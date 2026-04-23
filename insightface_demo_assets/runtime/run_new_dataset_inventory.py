import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

from dataset_profiles import discover_clip_pairs, load_dataset_profile_from_config
from scene_calibration import load_scene_calibration


def load_yaml_payload(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "offline_pipeline" in payload:
        return payload["offline_pipeline"]
    return payload


def resolve_path(project_root: Path, value: str):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def video_metadata(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = round((frame_count / fps), 3) if fps > 0 else 0.0
        return {
            "width": width,
            "height": height,
            "fps": round(fps, 4),
            "frame_count": frame_count,
            "duration_sec": duration_sec,
        }
    finally:
        cap.release()


def sample_person_counts(path: Path, model, sample_positions=None, conf=0.25, iou=0.35):
    sample_positions = sample_positions or [0.1, 0.3, 0.5, 0.7, 0.9]
    cap = cv2.VideoCapture(str(path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return {"sample_counts": [], "max_person_count": 0, "multi_subject_likely": False, "hard_scenario_likely": False}
        sample_indices = []
        for fraction in sample_positions:
            index = min(frame_count - 1, max(0, int(round(float(fraction) * frame_count))))
            if index not in sample_indices:
                sample_indices.append(index)
        rows = []
        max_person_count = 0
        for frame_index in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                rows.append({"frame_index": int(frame_index), "person_count": 0, "status": "decode_fail"})
                continue
            predictions = model.predict(frame, classes=[0], conf=conf, iou=iou, verbose=False, device="cpu")
            boxes = getattr(predictions[0], "boxes", None) if predictions else None
            person_count = 0 if boxes is None else len(boxes)
            rows.append({"frame_index": int(frame_index), "person_count": int(person_count), "status": "ok"})
            max_person_count = max(max_person_count, int(person_count))
        return {
            "sample_counts": rows,
            "max_person_count": int(max_person_count),
            "multi_subject_likely": bool(max_person_count > 1),
            "hard_scenario_likely": bool(max_person_count >= 3),
        }
    finally:
        cap.release()


def reference_calibration_for_physical_camera(dataset_profile, calibration, physical_camera_id):
    selected_cameras = list(dataset_profile.get("selected_cameras", []) or [])
    camera_cfgs = dataset_profile.get("cameras", {}) or {}
    calibration_cameras = calibration.get("cameras", {}) or {}
    candidate_ids = [
        camera_id
        for camera_id in selected_cameras
        if (camera_cfgs.get(camera_id, {}) or {}).get("source_physical_camera_id") == physical_camera_id
    ]
    candidate_ids.sort(key=lambda camera_id: bool((camera_cfgs.get(camera_id, {}) or {}).get("logical_demo_copy", False)))
    for camera_id in candidate_ids:
        if camera_id in calibration_cameras:
            return camera_id, calibration_cameras[camera_id]
    return "", {}


def calibration_reuse_assessment(dataset_profile, calibration, physical_camera_id, metadata, max_aspect_ratio_delta):
    calibration_camera_id, calibration_camera_cfg = reference_calibration_for_physical_camera(
        dataset_profile,
        calibration,
        physical_camera_id,
    )
    if not calibration_camera_id:
        return {
            "calibration_camera_id": "",
            "anchor_point_mode": "",
            "reference_width": 0,
            "reference_height": 0,
            "reference_aspect_ratio": 0.0,
            "clip_aspect_ratio": 0.0,
            "aspect_ratio_delta": 0.0,
            "reusable": False,
            "reason": "missing_calibration_source_camera",
        }
    frame_ref = calibration_camera_cfg.get("frame_size_ref", {}) or {}
    ref_width = int(frame_ref.get("width", 0) or 0)
    ref_height = int(frame_ref.get("height", 0) or 0)
    clip_width = int(metadata.get("width", 0) or 0)
    clip_height = int(metadata.get("height", 0) or 0)
    ref_aspect = (float(ref_width) / float(ref_height)) if ref_height > 0 else 0.0
    clip_aspect = (float(clip_width) / float(clip_height)) if clip_height > 0 else 0.0
    aspect_ratio_delta = abs(ref_aspect - clip_aspect)
    height_match = ref_height > 0 and clip_height == ref_height
    reusable = bool(height_match and aspect_ratio_delta <= max_aspect_ratio_delta)
    if reusable:
        reason = "same_source_camera_with_normalized_coordinates_and_compatible_aspect_ratio"
    elif not height_match:
        reason = "height_mismatch_requires_manual_review"
    else:
        reason = "aspect_ratio_delta_too_large_requires_manual_review"
    return {
        "calibration_camera_id": calibration_camera_id,
        "anchor_point_mode": str(calibration_camera_cfg.get("anchor_point_mode", "bottom_center") or "bottom_center"),
        "reference_width": ref_width,
        "reference_height": ref_height,
        "reference_aspect_ratio": round(ref_aspect, 6),
        "clip_aspect_ratio": round(clip_aspect, 6),
        "aspect_ratio_delta": round(aspect_ratio_delta, 6),
        "reusable": reusable,
        "reason": reason,
    }


def render_markdown(summary, pair_rows, output_path: Path):
    lines = []
    lines.append("# New Dataset Inventory")
    lines.append("")
    lines.append(f"- total_paired_clips_available: `{summary['total_paired_clips_available']}`")
    lines.append(f"- paired_clips_ready_for_eval: `{summary['paired_clips_ready_for_eval']}`")
    lines.append(f"- missing_required_clips_for_target_5: `{summary['missing_required_clips_for_target_5']}`")
    lines.append(f"- multi_subject_clip_count: `{summary['multi_subject_clip_count']}`")
    lines.append(f"- hard_scenario_clip_count: `{summary['hard_scenario_clip_count']}`")
    lines.append("")
    lines.append("| Pair | Ready | Multi-subject | Hard scenario | CAM1 reuse | CAM2 reuse |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in pair_rows:
        lines.append(
            f"| `{row['pair_id']}` | `{str(row['ready_for_eval']).lower()}` | "
            f"`{str(row['multi_subject_likely']).lower()}` | `{str(row['hard_scenario_likely']).lower()}` | "
            f"`{str(row['camera_1']['calibration_reuse']['reusable']).lower()}` | "
            f"`{str(row['camera_2']['calibration_reuse']['reusable']).lower()}` |"
        )
    lines.append("")
    if summary.get("missing_pair_ids"):
        lines.append("## Missing Pairs")
        lines.append("")
        for value in summary["missing_pair_ids"]:
            lines.append(f"- `{value}`")
        lines.append("")
    if summary.get("blockers"):
        lines.append("## Blockers")
        lines.append("")
        for value in summary["blockers"]:
            lines.append(f"- {value}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Inventory paired New Dataset clips and calibration reuse feasibility.")
    parser.add_argument(
        "--pipeline-config",
        default="insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml",
    )
    parser.add_argument("--output-dir", default="outputs/evaluations/new_dataset_inventory")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--person-conf-threshold", type=float, default=0.25)
    parser.add_argument("--person-iou-threshold", type=float, default=0.35)
    parser.add_argument("--max-aspect-ratio-delta-for-auto-reuse", type=float, default=0.12)
    args = parser.parse_args()

    pipeline_config_path = Path(args.pipeline_config).resolve()
    offline_config = load_yaml_payload(pipeline_config_path)
    project_root = resolve_path(pipeline_config_path.parent, offline_config.get("project_root", "."))
    dataset_profile_path, dataset_profile, dataset_profile_runtime = load_dataset_profile_from_config(
        offline_config,
        project_root,
        base_dir=pipeline_config_path.parent,
    )
    scene_calibration_path = resolve_path(project_root, offline_config.get("scene_calibration_config", ""))
    calibration, calibration_runtime = load_scene_calibration(
        config_path=scene_calibration_path,
        base_dir=pipeline_config_path.parent,
        required=True,
        camera_ids=dataset_profile.get("selected_cameras", []),
    )

    discovered = discover_clip_pairs(dataset_profile, project_root)
    available_pair_ids = list(discovered.get("available_pair_ids", []))
    files_by_camera = discovered.get("files_by_camera", {})
    physical_cameras = dataset_profile.get("physical_cameras", {}) or {}

    from ultralytics import YOLO

    detector_model = ((offline_config.get("multi_source_inference", {}) or {}).get("detector_model", "") or "yolov8n.pt").strip()
    detector = YOLO(detector_model)
    sample_positions = [round((index + 1) / float(args.sample_count + 1), 3) for index in range(max(1, args.sample_count))]

    camera_file_rows = defaultdict(dict)
    stems_by_camera = {}
    for physical_camera_id, physical_cfg in physical_cameras.items():
        label = physical_cfg.get("label", physical_camera_id)
        file_map = files_by_camera.get(physical_camera_id, {})
        stems_by_camera[physical_camera_id] = sorted(file_map.keys())
        for stem, file_path in sorted(file_map.items()):
            metadata = video_metadata(file_path)
            people_summary = sample_person_counts(
                file_path,
                detector,
                sample_positions=sample_positions,
                conf=args.person_conf_threshold,
                iou=args.person_iou_threshold,
            )
            calibration_reuse = calibration_reuse_assessment(
                dataset_profile,
                calibration,
                physical_camera_id,
                metadata,
                max_aspect_ratio_delta=args.max_aspect_ratio_delta_for_auto_reuse,
            )
            camera_file_rows[physical_camera_id][stem] = {
                "physical_camera_id": physical_camera_id,
                "camera_label": label,
                "stem": stem,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "metadata": metadata,
                "sampled_person_audit": people_summary,
                "calibration_reuse": calibration_reuse,
            }

    pair_rows = []
    multi_subject_stems = []
    hard_scenario_stems = []
    ready_for_eval = []
    for pair_id in available_pair_ids:
        cam1 = camera_file_rows.get("CAM1", {}).get(pair_id)
        cam2 = camera_file_rows.get("CAM2", {}).get(pair_id)
        if not cam1 or not cam2:
            continue
        pair_multi_subject = bool(
            cam1["sampled_person_audit"]["multi_subject_likely"]
            or cam2["sampled_person_audit"]["multi_subject_likely"]
        )
        pair_hard = bool(
            cam1["sampled_person_audit"]["hard_scenario_likely"]
            or cam2["sampled_person_audit"]["hard_scenario_likely"]
        )
        pair_ready = bool(
            cam1["calibration_reuse"]["reusable"]
            and cam2["calibration_reuse"]["reusable"]
        )
        if pair_multi_subject:
            multi_subject_stems.append(pair_id)
        if pair_hard:
            hard_scenario_stems.append(pair_id)
        if pair_ready:
            ready_for_eval.append(pair_id)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "ready_for_eval": pair_ready,
                "multi_subject_likely": pair_multi_subject,
                "hard_scenario_likely": pair_hard,
                "camera_1": cam1,
                "camera_2": cam2,
            }
        )

    all_stems = sorted(set(stem for stems in stems_by_camera.values() for stem in stems))
    missing_pair_ids = sorted(set(all_stems) - set(available_pair_ids))
    blockers = []
    if len(available_pair_ids) < 5:
        blockers.append(
            f"Only {len(available_pair_ids)} paired clips are available locally; supervisor target requires at least 5 paired clips."
        )
    if not multi_subject_stems:
        blockers.append("No multi-subject clip was detected in the sampled inventory audit.")

    summary = {
        "pipeline_config_path": str(pipeline_config_path),
        "project_root": str(project_root),
        "dataset_profile_runtime": dataset_profile_runtime,
        "scene_calibration_runtime": calibration_runtime,
        "total_paired_clips_available": len(available_pair_ids),
        "paired_clips_ready_for_eval": len(ready_for_eval),
        "paired_clip_ids": available_pair_ids,
        "ready_for_eval_pair_ids": ready_for_eval,
        "missing_pair_ids": missing_pair_ids,
        "required_clip_target": 5,
        "missing_required_clips_for_target_5": max(0, 5 - len(available_pair_ids)),
        "multi_subject_clip_count": len(multi_subject_stems),
        "multi_subject_clip_ids": multi_subject_stems,
        "hard_scenario_clip_count": len(hard_scenario_stems),
        "hard_scenario_clip_ids": hard_scenario_stems,
        "blockers": blockers,
        "per_camera_stems": stems_by_camera,
    }
    inventory_payload = {
        "summary": summary,
        "pair_rows": pair_rows,
        "per_camera_inventory": camera_file_rows,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_inventory.json").write_text(
        json.dumps(inventory_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_markdown(summary, pair_rows, output_dir / "dataset_inventory.md")
    (output_dir / "calibration_reuse_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "pair_reuse": [
                    {
                        "pair_id": row["pair_id"],
                        "camera_1_reuse": row["camera_1"]["calibration_reuse"],
                        "camera_2_reuse": row["camera_2"]["calibration_reuse"],
                    }
                    for row in pair_rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"PAIRED_CLIPS={len(available_pair_ids)}")
    print(f"READY_FOR_EVAL={len(ready_for_eval)}")
    print(output_dir / "dataset_inventory.json")


if __name__ == "__main__":
    main()
