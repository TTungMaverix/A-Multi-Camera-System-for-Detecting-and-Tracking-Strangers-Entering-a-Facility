import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import yaml

from association_core import load_camera_transition_map
from offline_pipeline.event_builder import build_offline_stage_inputs, load_json, save_json
from run_face_resolution_demo import main as run_face_resolution_main
from scene_calibration import (
    apply_scene_calibration_to_transition_map,
    apply_scene_calibration_to_wildtrack_config,
    load_runtime_scene_calibration,
)


def load_pipeline_config(config_path: Path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "offline_pipeline" in payload:
        return payload["offline_pipeline"]
    return payload


def resolve_path(project_root: Path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def build_face_runtime_config(offline_config, stage_inputs, output_root: Path):
    project_root = Path(stage_inputs["project_root"]).resolve()
    face_demo_config_path = resolve_path(project_root, offline_config["face_demo_config"])
    face_demo_config = load_json(face_demo_config_path)
    runtime_dir = output_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = dict(face_demo_config)
    runtime_config.update(
        {
            "project_root": str(project_root),
            "wildtrack_config_path": stage_inputs["wildtrack_config_path"],
            "tracks_csv": stage_inputs["tracks_csv"],
            "wildtrack_identity_queue_csv": stage_inputs["identity_queue_csv"],
            "resolved_events_csv": str(runtime_dir / "resolved_events_template.csv"),
            "unknown_profiles_csv": str(runtime_dir / "unknown_profiles_template.csv"),
            "known_face_embeddings_csv": str(runtime_dir / "known_face_embeddings_template.csv"),
        }
    )
    if offline_config.get("face_demo_overrides"):
        runtime_config.update(dict(offline_config["face_demo_overrides"]))
    if offline_config.get("association_policy_config"):
        runtime_config["association_policy_config"] = str(
            resolve_path(project_root, offline_config["association_policy_config"])
        )
    if offline_config.get("camera_transition_map_config"):
        runtime_config["camera_transition_map_config"] = str(
            resolve_path(project_root, offline_config["camera_transition_map_config"])
        )
    if offline_config.get("scene_calibration_config"):
        runtime_config["scene_calibration_config"] = str(
            resolve_path(project_root, offline_config["scene_calibration_config"])
        )
    if offline_config.get("known_gallery", {}).get("manifest_csv"):
        runtime_config["known_face_manifest_csv"] = str(
            resolve_path(project_root, offline_config["known_gallery"]["manifest_csv"])
        )
    if offline_config.get("known_gallery", {}).get("gallery_root"):
        runtime_config["known_face_gallery_root"] = str(
            resolve_path(project_root, offline_config["known_gallery"]["gallery_root"])
        )
    runtime_config_path = runtime_dir / "face_demo_runtime_config.json"
    runtime_config_path.write_text(json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return runtime_config_path, runtime_config


def build_source_lookup(project_root: Path, offline_config):
    lookup = {}
    for camera_id, video_path in (offline_config.get("dataset", {}).get("video_sources", {}) or {}).items():
        lookup[camera_id] = {
            "source_type": "file",
            "source": str(resolve_path(project_root, video_path)),
        }
    return lookup


def sync_final_outputs(output_root: Path):
    runtime_dir = output_root / "runtime"
    events_dir = output_root / "events"
    timelines_dir = output_root / "timelines"
    summaries_dir = output_root / "summaries"
    audit_dir = output_root / "audit"
    association_logs_dir = output_root / "association_logs"
    for directory in (events_dir, timelines_dir, summaries_dir, audit_dir, association_logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copy_pairs = [
        (runtime_dir / "resolved_events_template.csv", events_dir / "resolved_events.csv"),
        (runtime_dir / "stream_identity_timeline.csv", timelines_dir / "stream_identity_timeline.csv"),
        (runtime_dir / "unknown_identity_timeline.csv", timelines_dir / "unknown_identity_timeline.csv"),
        (runtime_dir / "unknown_identity_timeline.json", timelines_dir / "unknown_identity_timeline.json"),
        (runtime_dir / "unknown_profiles_template.csv", timelines_dir / "unknown_profiles.csv"),
        (runtime_dir / "face_resolution_summary.json", summaries_dir / "face_resolution_summary.json"),
        (runtime_dir / "face_body_usage_summary.json", summaries_dir / "face_body_usage_summary.json"),
    ]
    for src, dst in copy_pairs:
        if src.exists():
            shutil.copy2(src, dst)

    for audit_file in runtime_dir.glob("audit_*"):
        if audit_file.is_file():
            shutil.copy2(audit_file, audit_dir / audit_file.name)
    runtime_association_dir = runtime_dir / "association_logs"
    if runtime_association_dir.exists():
        for path in runtime_association_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, association_logs_dir / path.name)


def export_unknown_id_mapping(output_root: Path):
    resolved_events_path = output_root / "events" / "resolved_events.csv"
    if not resolved_events_path.exists():
        return ""
    with resolved_events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping_rows = []
    for row in rows:
        if row.get("identity_status") != "unknown" or not row.get("unknown_global_id"):
            continue
        mapping_rows.append(
            {
                "event_id": row.get("event_id", ""),
                "camera_id": row.get("camera_id", ""),
                "frame_id": row.get("frame_id", ""),
                "relative_sec": row.get("relative_sec", ""),
                "global_gt_id_for_audit": row.get("global_gt_id", ""),
                "anchor_camera_id": row.get("anchor_camera_id", ""),
                "anchor_relative_sec": row.get("anchor_relative_sec", ""),
                "relation_type": row.get("relation_type", ""),
                "unknown_global_id": row.get("unknown_global_id", ""),
                "resolved_global_id": row.get("resolved_global_id", ""),
                "resolution_source": row.get("resolution_source", ""),
                "decision_reason": row.get("decision_reason", ""),
                "reason_code": row.get("reason_code", ""),
                "best_head_crop": row.get("best_head_crop", ""),
                "best_body_crop": row.get("best_body_crop", ""),
            }
        )
    output_path = output_root / "events" / "unknown_id_mapping.csv"
    if not mapping_rows:
        output_path.write_text("event_id,camera_id,relative_sec,unknown_global_id,resolved_global_id\r\n", encoding="utf-8-sig")
        return str(output_path)
    fieldnames = list(mapping_rows[0].keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(row)
    return str(output_path)


def run_offline_pipeline(config_path: Path, cli_overrides=None):
    pipeline_started = time.perf_counter()
    cli_overrides = cli_overrides or {}
    offline_config = load_pipeline_config(config_path)
    execution_mode = (offline_config.get("execution", {}) or {}).get("mode", "sequential")
    if execution_mode == "multiprocessing":
        from offline_pipeline.multiprocessing_runner import run_offline_pipeline_multiprocess

        return run_offline_pipeline_multiprocess(config_path, cli_overrides=cli_overrides)
    project_root = resolve_path(config_path.parent, offline_config.get("project_root", "."))
    offline_config["project_root"] = str(project_root)

    if cli_overrides.get("association_policy_config"):
        offline_config["association_policy_config"] = cli_overrides["association_policy_config"]
    if cli_overrides.get("camera_transition_map_config"):
        offline_config["camera_transition_map_config"] = cli_overrides["camera_transition_map_config"]
    if cli_overrides.get("known_manifest_csv"):
        offline_config.setdefault("known_gallery", {})
        offline_config["known_gallery"]["manifest_csv"] = cli_overrides["known_manifest_csv"]
    if cli_overrides.get("known_gallery_root"):
        offline_config.setdefault("known_gallery", {})
        offline_config["known_gallery"]["gallery_root"] = cli_overrides["known_gallery_root"]
    if cli_overrides.get("video_sources"):
        offline_config.setdefault("dataset", {})
        offline_config["dataset"]["video_sources"] = cli_overrides["video_sources"]

    output_root = resolve_path(project_root, offline_config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    wildtrack_config_path = resolve_path(project_root, offline_config["wildtrack_demo_config"])
    wildtrack_config = load_json(wildtrack_config_path)
    transition_config_path = offline_config.get("camera_transition_map_config", "")
    transition_map, transition_runtime = load_camera_transition_map(
        wildtrack_config,
        config_path=str(resolve_path(project_root, transition_config_path)) if transition_config_path else "",
        base_dir=config_path.parent,
    )
    scene_calibration_config = offline_config.get("scene_calibration_config", "")
    scene_calibration, _runtime_cameras, scene_runtime = load_runtime_scene_calibration(
        config_path=resolve_path(project_root, scene_calibration_config) if scene_calibration_config else None,
        base_dir=config_path.parent,
        camera_ids=wildtrack_config.get("selected_cameras", []),
        source_lookup=build_source_lookup(project_root, offline_config),
        required=True,
    )
    frame_sizes = scene_runtime.get("frame_sizes", {})
    transition_map = apply_scene_calibration_to_transition_map(transition_map, scene_calibration, frame_sizes)
    wildtrack_config = apply_scene_calibration_to_wildtrack_config(wildtrack_config, scene_calibration, frame_sizes)

    stage_started = time.perf_counter()
    stage_inputs = build_offline_stage_inputs(offline_config, transition_map, wildtrack_config_override=wildtrack_config)
    stage_elapsed_sec = round(max(time.perf_counter() - stage_started, 1e-9), 3)
    runtime_config_path, runtime_config = build_face_runtime_config(offline_config, stage_inputs, output_root)
    face_resolution_started = time.perf_counter()
    run_face_resolution_main(runtime_config_path)
    face_resolution_elapsed_sec = round(max(time.perf_counter() - face_resolution_started, 1e-9), 3)
    sync_started = time.perf_counter()
    sync_final_outputs(output_root)
    sync_elapsed_sec = round(max(time.perf_counter() - sync_started, 1e-9), 3)
    unknown_id_mapping_csv = export_unknown_id_mapping(output_root)

    pipeline_summary = {
        "pipeline_name": offline_config.get("pipeline_name", "offline_multicam_pipeline"),
        "source_backend": offline_config.get("source_backend", "wildtrack_gt_annotations"),
        "execution_mode": offline_config.get("execution", {}).get("mode", "sequential"),
        "config_path": str(config_path),
        "output_root": str(output_root),
        "stage_inputs": stage_inputs,
        "runtime_config_path": str(runtime_config_path),
        "camera_transition_map_runtime": transition_runtime,
        "scene_calibration_runtime": scene_runtime,
        "face_runtime_config": {
            "association_policy_config": runtime_config.get("association_policy_config", ""),
            "camera_transition_map_config": runtime_config.get("camera_transition_map_config", ""),
            "scene_calibration_config": runtime_config.get("scene_calibration_config", ""),
            "known_face_manifest_csv": runtime_config.get("known_face_manifest_csv", ""),
            "known_face_gallery_root": runtime_config.get("known_face_gallery_root", ""),
        },
        "final_outputs": {
            "resolved_events_csv": str(output_root / "events" / "resolved_events.csv"),
            "unknown_id_mapping_csv": unknown_id_mapping_csv,
            "stream_identity_timeline_csv": str(output_root / "timelines" / "stream_identity_timeline.csv"),
            "unknown_identity_timeline_csv": str(output_root / "timelines" / "unknown_identity_timeline.csv"),
            "unknown_identity_timeline_json": str(output_root / "timelines" / "unknown_identity_timeline.json"),
            "summary_json": str(output_root / "summaries" / "face_resolution_summary.json"),
            "face_body_usage_summary_json": str(output_root / "summaries" / "face_body_usage_summary.json"),
            "association_logs_dir": str(output_root / "association_logs"),
            "audit_dir": str(output_root / "audit"),
        },
        "timings_sec": {
            "stage_input_build_sec": stage_elapsed_sec,
            "face_resolution_sec": face_resolution_elapsed_sec,
            "final_sync_sec": sync_elapsed_sec,
            "total_pipeline_sec": round(max(time.perf_counter() - pipeline_started, 1e-9), 3),
        },
        "architecture_audit": {
            "execution_mode": offline_config.get("execution", {}).get("mode", "sequential"),
            "architecture_mode": "synchronous_replay_pipeline",
            "notes": [
                "The current Cam6 replay debug path is sequential: stage inputs are built first, then face/body/association resolution runs, then outputs are synchronized.",
                "The live pipeline keeps a separate multiprocessing producer-consumer design and reports its own worker/consumer FPS in live_pipeline_summary.json.",
            ],
        },
    }
    save_json(output_root / "summaries" / "offline_pipeline_summary.json", pipeline_summary)
    return pipeline_summary


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run the offline multi-camera stranger demo pipeline.")
    parser.add_argument("--config", required=True, help="Path to offline pipeline config YAML.")
    parser.add_argument("--association-policy-config", default="", help="Override association policy config path.")
    parser.add_argument("--camera-transition-map-config", default="", help="Override camera transition map config path.")
    parser.add_argument("--known-manifest-csv", default="", help="Override known face manifest CSV.")
    parser.add_argument("--known-gallery-root", default="", help="Override known face gallery root.")
    parser.add_argument(
        "--video-source",
        action="append",
        default=[],
        help="Override video source in the form CAMERA_ID=PATH. Repeat for multiple cameras.",
    )
    return parser.parse_args()


def cli():
    args = parse_cli_args()
    video_sources = {}
    for item in args.video_source:
        if "=" not in item:
            raise SystemExit(f"Invalid --video-source value: {item}")
        camera_id, value = item.split("=", 1)
        video_sources[camera_id.strip()] = value.strip()
    run_offline_pipeline(
        Path(args.config).resolve(),
        cli_overrides={
            "association_policy_config": args.association_policy_config,
            "camera_transition_map_config": args.camera_transition_map_config,
            "known_manifest_csv": args.known_manifest_csv,
            "known_gallery_root": args.known_gallery_root,
            "video_sources": video_sources,
        },
    )


if __name__ == "__main__":
    cli()
