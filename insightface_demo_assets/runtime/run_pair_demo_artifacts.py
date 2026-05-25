import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

from dataset_profiles import load_dataset_profile_from_config
from offline_pipeline.orchestrator import load_pipeline_config, resolve_path, run_offline_pipeline
from run_live_event_demo_server import (
    load_association_summary,
    load_face_body_usage_summary,
    load_face_resolution_summary,
    load_identity_timeline,
    load_known_db_summary,
    load_latest_events,
    load_offline_pipeline_summary,
    load_reid_handoffs,
)
from run_new_dataset_evaluation import build_temp_pipeline_config


DEFAULT_PIPELINE_CONFIG = "insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml"
DEFAULT_KNOWN_MANIFEST = "insightface_demo_assets/known_face_facility_manifest.csv"


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline pipeline artifacts for one demo pair.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--pipeline-config", default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-calibration-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--known-db-root", required=True)
    parser.add_argument("--known-manifest-csv", default=DEFAULT_KNOWN_MANIFEST)
    return parser.parse_args()


def build_temp_dataset_profile(
    base_config,
    project_root: Path,
    pipeline_config_path: Path,
    dataset_root: Path,
    known_db_root: Path,
):
    profile_path, dataset_profile, dataset_profile_runtime = load_dataset_profile_from_config(
        base_config,
        project_root,
        base_dir=pipeline_config_path.parent,
    )
    profile = deepcopy(dataset_profile)
    profile["dataset_root"] = str(dataset_root)
    profile.setdefault("known_db", {})
    profile["known_db"]["enabled"] = True
    profile["known_db"]["path"] = str(known_db_root)
    for physical_camera_id, folder_name in (("CAM1", "Camera 1"), ("CAM2", "Camera 2")):
        if physical_camera_id in profile.get("physical_cameras", {}):
            profile["physical_cameras"][physical_camera_id]["root_dir"] = str(dataset_root / folder_name)
    return profile_path, profile, dataset_profile_runtime


def summarize_run(project_root: Path, output_root: Path, pair_id: str):
    latest_events = load_latest_events(output_root)
    timeline_rows = load_identity_timeline(output_root)
    handoff_rows = load_reid_handoffs(output_root)
    face_resolution_summary = load_face_resolution_summary(output_root)
    face_body_usage_summary = load_face_body_usage_summary(output_root)
    association_summary = load_association_summary(output_root)
    offline_pipeline_summary = load_offline_pipeline_summary(output_root)
    face_metrics = face_body_usage_summary.get("metrics", face_body_usage_summary) if isinstance(face_body_usage_summary, dict) else {}
    mode_b = face_resolution_summary.get("mode_b_true_assoc", {}) if isinstance(face_resolution_summary, dict) else {}
    known_db_runtime = face_resolution_summary.get("known_db_runtime", {}) if isinstance(face_resolution_summary, dict) else {}
    fallback_known_db = load_known_db_summary(project_root)
    metrics = association_summary.get("metrics", association_summary) if isinstance(association_summary, dict) else {}
    summary = {
        "pair_id": pair_id,
        "output_root": str(output_root),
        "event_count": len(latest_events),
        "timeline_identity_count": len(timeline_rows),
        "handoff_count": len(handoff_rows),
        "face_candidate_count": face_metrics.get("face_candidate_count", "KEY_NOT_FOUND"),
        "face_embedding_created_count": face_metrics.get("face_embedding_created_count", "KEY_NOT_FOUND"),
        "known_match_success_count": face_metrics.get("known_face_match_success_count", "KEY_NOT_FOUND"),
        "known_db_identity_count": known_db_runtime.get("identities_loaded", fallback_known_db.get("identity_count", 0)),
        "known_db_embedding_count": known_db_runtime.get("embedding_count", fallback_known_db.get("embedding_count", 0)),
        "known_event_count": mode_b.get("known_event_count", "KEY_NOT_FOUND"),
        "unknown_event_count": mode_b.get("unknown_event_count", "KEY_NOT_FOUND"),
        "unknown_reuse_count": metrics.get("unknown_reuse_count", mode_b.get("unknown_reuse_count", "KEY_NOT_FOUND")),
        "new_unknown_count": metrics.get("new_unknown_count", mode_b.get("new_unknown_count", "KEY_NOT_FOUND")),
        "total_pipeline_sec": (offline_pipeline_summary.get("timings_sec", {}) or {}).get("total_pipeline_sec", "KEY_NOT_FOUND"),
    }
    return summary


def main():
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    pipeline_config_path = resolve_path(project_root, args.pipeline_config)
    dataset_root = Path(args.dataset_root).resolve()
    known_db_root = Path(args.known_db_root).resolve()
    output_root = resolve_path(project_root, args.output_root)
    scene_calibration_config = resolve_path(project_root, args.scene_calibration_config)
    known_manifest_csv = resolve_path(project_root, args.known_manifest_csv)

    base_config = load_pipeline_config(pipeline_config_path)
    temp_config = build_temp_pipeline_config(base_config, args.pair_id, output_root, project_root)["offline_pipeline"]
    temp_config["pipeline_name"] = f"demo_pair_{args.pair_id}"
    temp_config["scene_calibration_config"] = str(scene_calibration_config)
    temp_config["output_root"] = str(output_root)
    temp_config.setdefault("low_load", {})
    temp_config["low_load"]["enabled"] = False
    temp_config.setdefault("known_gallery", {})
    temp_config["known_gallery"]["manifest_csv"] = str(known_manifest_csv)
    temp_config["known_gallery"]["gallery_root"] = str(known_db_root)

    _profile_path, temp_dataset_profile, _profile_runtime = build_temp_dataset_profile(
        base_config,
        project_root,
        pipeline_config_path,
        dataset_root,
        known_db_root,
    )
    runtime_dir = output_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    temp_dataset_profile_path = runtime_dir / f"{args.pair_id}_dataset_profile.runtime.yaml"
    temp_dataset_profile_path.write_text(
        yaml.safe_dump({"dataset_profile": temp_dataset_profile}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temp_config["dataset_profile_config"] = str(temp_dataset_profile_path)

    temp_config_path = runtime_dir / f"{args.pair_id}_offline_pipeline.runtime.yaml"
    temp_config_path.write_text(
        yaml.safe_dump({"offline_pipeline": temp_config}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"PAIR_ID={args.pair_id}")
    print(f"PIPELINE_CONFIG={pipeline_config_path}")
    print(f"DATASET_ROOT={dataset_root}")
    print(f"SCENE_CALIBRATION_CONFIG={scene_calibration_config}")
    print(f"OUTPUT_ROOT={output_root}")
    print(f"KNOWN_DB_ROOT={known_db_root}")
    print(f"TEMP_DATASET_PROFILE={temp_dataset_profile_path}")
    print(f"TEMP_PIPELINE_CONFIG={temp_config_path}")

    pipeline_summary = run_offline_pipeline(temp_config_path)
    save_json(output_root / "summaries" / "demo_artifact_generation_summary.json", pipeline_summary)
    artifact_summary = summarize_run(project_root, output_root, args.pair_id)
    print("ARTIFACT_RUN_SUMMARY=" + json.dumps(artifact_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
