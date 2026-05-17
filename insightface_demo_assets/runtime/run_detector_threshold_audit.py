import argparse
import copy
import json
from pathlib import Path

import yaml

from association_core import load_camera_transition_map
from dataset_profiles import load_dataset_profile_from_config
from offline_pipeline.event_builder import build_offline_stage_inputs
from offline_pipeline.orchestrator import build_source_lookup, load_pipeline_config, resolve_path
from scene_calibration import (
    apply_scene_calibration_to_transition_map,
    apply_scene_calibration_to_wildtrack_config,
    load_runtime_scene_calibration,
)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_entry_events(events_json: Path):
    if not events_json.exists():
        return {"total_entry_in_events": 0, "per_camera_entry_in_events": {}}
    payload = json.loads(events_json.read_text(encoding="utf-8"))
    per_camera = {}
    for row in payload:
        camera_id = str(row.get("camera_id", "") or "")
        per_camera[camera_id] = per_camera.get(camera_id, 0) + 1
    return {
        "total_entry_in_events": len(payload),
        "per_camera_entry_in_events": per_camera,
    }


def _load_stage_summary(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_runtime_config(base_config_path: Path, pair_id: str, threshold: float, output_dir: Path):
    offline_config = load_pipeline_config(base_config_path)
    project_root = resolve_path(base_config_path.parent, offline_config.get("project_root", "."))
    offline_config["project_root"] = str(project_root)
    offline_config.setdefault("logical_demo", {})
    offline_config["logical_demo"]["pair_id"] = pair_id
    offline_config.setdefault("multi_source_inference", {})
    offline_config["multi_source_inference"]["conf_threshold"] = float(threshold)
    cache_cfg = offline_config["multi_source_inference"].setdefault("cache", {})
    cache_cfg["enabled"] = False
    cache_cfg["use_cache"] = False
    cache_cfg["refresh_cache"] = True
    offline_config["multi_source_inference"]["cache"] = cache_cfg
    offline_config["output_root"] = str((output_dir / f"{pair_id}_conf_{str(threshold).replace('.', '_')}").resolve())
    return project_root, offline_config


def _build_stage_inputs(base_config_path: Path, pair_id: str, threshold: float, output_dir: Path):
    project_root, offline_config = _prepare_runtime_config(base_config_path, pair_id, threshold, output_dir)
    dataset_profile_path, dataset_profile, _dataset_profile_runtime = load_dataset_profile_from_config(
        offline_config,
        project_root,
        base_dir=base_config_path.parent,
    )
    transition_config_path = offline_config.get("camera_transition_map_config", "")
    transition_map, _transition_runtime = load_camera_transition_map(
        dataset_profile,
        config_path=str(resolve_path(project_root, transition_config_path)) if transition_config_path else "",
        base_dir=base_config_path.parent,
    )
    scene_calibration_config = offline_config.get("scene_calibration_config", "")
    scene_calibration, _runtime_cameras, scene_runtime = load_runtime_scene_calibration(
        config_path=resolve_path(project_root, scene_calibration_config) if scene_calibration_config else None,
        base_dir=base_config_path.parent,
        camera_ids=dataset_profile.get("selected_cameras", []),
        source_lookup=build_source_lookup(project_root, offline_config),
        required=True,
    )
    frame_sizes = scene_runtime.get("frame_sizes", {})
    transition_map = apply_scene_calibration_to_transition_map(transition_map, scene_calibration, frame_sizes)
    dataset_profile = apply_scene_calibration_to_wildtrack_config(dataset_profile, scene_calibration, frame_sizes)
    build_offline_stage_inputs(offline_config, transition_map, dataset_profile_override=dataset_profile)
    run_output_dir = Path(offline_config["output_root"])
    stage_summary = _load_stage_summary(run_output_dir / "summaries" / "stage_input_summary.json")
    event_summary = _count_entry_events(run_output_dir / "events" / "entry_in_events.json")
    per_physical_runtime = (
        stage_summary.get("multi_source_inference", {}).get("per_physical_camera_runtime", {})
        or stage_summary.get("multi_source_inference", {}).get("per_camera_runtime", {})
        or {}
    )
    return {
        "pair_id": pair_id,
        "conf_threshold": float(threshold),
        "output_root": str(run_output_dir),
        "filtered_track_rows": int(stage_summary.get("filtered_track_rows", 0) or 0),
        "per_camera_rows": stage_summary.get("per_camera_rows", {}),
        "per_physical_camera_runtime": per_physical_runtime,
        "total_entry_in_events": event_summary["total_entry_in_events"],
        "per_camera_entry_in_events": event_summary["per_camera_entry_in_events"],
        "notes": [
            "This audit measures detector-threshold side effects on predicted person boxes, track rows, and ENTRY_IN events.",
            "False-positive / false-negative judgement still requires manual overlay review; this script only reports the artifact counts."
        ],
    }


def _recommend(results):
    if not results:
        return {"recommended_threshold": None, "reason": "no_results"}
    baseline = results[0]
    baseline_events = int(baseline.get("total_entry_in_events", 0) or 0)
    baseline_rows = int(baseline.get("filtered_track_rows", 0) or 0)
    recommended = baseline["conf_threshold"]
    reason = "keep_current_threshold"
    for row in results[1:]:
        events = int(row.get("total_entry_in_events", 0) or 0)
        tracks = int(row.get("filtered_track_rows", 0) or 0)
        if events >= baseline_events and tracks >= int(round(baseline_rows * 0.75)):
            recommended = row["conf_threshold"]
            reason = "higher_threshold_kept_events_without_large_track_loss"
    return {
        "recommended_threshold": recommended,
        "reason": reason,
        "baseline_threshold": baseline["conf_threshold"],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit detector confidence thresholds on a selected New Dataset pair.")
    parser.add_argument(
        "--base-config",
        default="insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml",
    )
    parser.add_argument("--pair-id", default="a3")
    parser.add_argument("--thresholds", default="0.4,0.5,0.6")
    parser.add_argument("--output-dir", default="outputs/evaluations/detector_threshold_audit")
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(item.strip()) for item in str(args.thresholds).split(",") if str(item).strip()]
    rows = [_build_stage_inputs(base_config_path, args.pair_id, threshold, output_dir) for threshold in thresholds]
    summary = {
        "pair_id": args.pair_id,
        "base_config": str(base_config_path),
        "results": rows,
        "recommendation": _recommend(rows),
    }
    save_json(output_dir / f"{args.pair_id}_detector_threshold_audit.json", summary)
    print(output_dir / f"{args.pair_id}_detector_threshold_audit.json")


if __name__ == "__main__":
    main()
