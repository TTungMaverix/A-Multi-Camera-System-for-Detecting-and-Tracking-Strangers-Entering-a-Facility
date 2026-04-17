import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

from association_core import load_camera_transition_map
from evaluation_utils import (
    compute_event_level_idf1,
    compute_local_tracking_mota,
    extract_aligned_gt_rows_from_annotations,
    match_event_rows_to_gt,
    normalize_identity_id,
    read_csv_rows,
    safe_float,
    safe_int,
    save_json,
    write_csv_rows,
)
from offline_pipeline.event_builder import build_offline_stage_inputs, load_json
from offline_pipeline.orchestrator import load_pipeline_config
from scene_calibration import (
    apply_scene_calibration_to_transition_map,
    apply_scene_calibration_to_wildtrack_config,
    load_runtime_scene_calibration,
)


def resolve_path(base_dir: Path, value):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_eval_config(config_path: Path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return payload.get("quantitative_evaluation", payload)


def build_source_lookup(project_root: Path, offline_config):
    lookup = {}
    for camera_id, video_path in (offline_config.get("dataset", {}).get("video_sources", {}) or {}).items():
        lookup[camera_id] = {
            "source_type": "file",
            "source": str(resolve_path(project_root, video_path)),
        }
    return lookup


def build_local_tracking_gt_rows(project_root: Path, benchmark_offline_config, pred_local_rows):
    replay_cfg = benchmark_offline_config.get("single_source_replay", {}) or {}
    source_camera_id = replay_cfg.get("source_camera_id", "C6")
    source_template_path = resolve_path(project_root, replay_cfg.get("source_template_config", "wildtrack_demo/wildtrack_demo_config.json"))
    source_template = load_json(source_template_path)
    source_camera_cfg = (source_template.get("cameras", {}) or {}).get(source_camera_id, {})
    view_index = safe_int(source_camera_cfg.get("view_index"), 5)
    dataset_root = resolve_path(project_root, benchmark_offline_config.get("dataset", {}).get("root", "Wildtrack"))
    annotation_dir = dataset_root / "annotations_positions"
    processing_polygon = None
    scene_cfg = benchmark_offline_config.get("scene_calibration_config", "")
    if scene_cfg:
        scene_payload = load_json(resolve_path(project_root, scene_cfg))
        camera_payload = (((scene_payload.get("scene_calibration", {}) or {}).get("cameras", {}) or {}).get("C1", {}))
        polygon = ((((camera_payload.get("processing_roi", {}) or {}).get("polygon")) or []))
        frame_ref = (camera_payload.get("frame_size_ref", {}) or {})
        frame_width = safe_float(frame_ref.get("width"), 1920.0)
        frame_height = safe_float(frame_ref.get("height"), 1080.0)
        if polygon:
            processing_polygon = [
                (safe_float(point[0]) * frame_width, safe_float(point[1]) * frame_height)
                for point in polygon
            ]
    logical_to_actual_frames = {}
    for row in pred_local_rows:
        logical_to_actual_frames[safe_int(row.get("frame_id"))] = safe_int(row.get("source_frame_id_actual", row.get("frame_id")))
    return extract_aligned_gt_rows_from_annotations(
        annotation_dir,
        view_index,
        logical_to_actual_frames,
        processing_polygon=processing_polygon,
    )


def _event_key(row):
    return (
        row.get("camera_id", ""),
        safe_int(row.get("frame_id")),
        round(safe_float(row.get("relative_sec")), 3),
    )


def build_gt_stage_inputs(project_root: Path, benchmark_config_path: Path, gt_output_root: Path):
    benchmark_offline_config = load_pipeline_config(benchmark_config_path)
    benchmark_offline_config["project_root"] = str(project_root)
    benchmark_offline_config["output_root"] = str(gt_output_root)
    benchmark_offline_config = copy.deepcopy(benchmark_offline_config)
    benchmark_offline_config.setdefault("single_source_replay", {})
    benchmark_offline_config["single_source_replay"]["track_provider"] = "gt_annotations"

    wildtrack_config_path = resolve_path(project_root, benchmark_offline_config["wildtrack_demo_config"])
    wildtrack_config = load_json(wildtrack_config_path)
    transition_cfg = benchmark_offline_config.get("camera_transition_map_config", "")
    transition_map, _transition_runtime = load_camera_transition_map(
        wildtrack_config,
        config_path=str(resolve_path(project_root, transition_cfg)) if transition_cfg else "",
        base_dir=benchmark_config_path.parent,
    )
    scene_cfg = benchmark_offline_config.get("scene_calibration_config", "")
    scene_calibration, _runtime_cameras, scene_runtime = load_runtime_scene_calibration(
        config_path=resolve_path(project_root, scene_cfg) if scene_cfg else None,
        base_dir=benchmark_config_path.parent,
        camera_ids=wildtrack_config.get("selected_cameras", []),
        source_lookup=build_source_lookup(project_root, benchmark_offline_config),
        required=True,
    )
    frame_sizes = scene_runtime.get("frame_sizes", {})
    transition_map = apply_scene_calibration_to_transition_map(transition_map, scene_calibration, frame_sizes)
    wildtrack_config = apply_scene_calibration_to_wildtrack_config(wildtrack_config, scene_calibration, frame_sizes)
    return build_offline_stage_inputs(benchmark_offline_config, transition_map, wildtrack_config_override=wildtrack_config)


def build_resolved_rows_for_event_matching(resolved_rows, queue_rows):
    queue_by_key = {_event_key(row): row for row in queue_rows}
    merged = []
    for row in resolved_rows:
        merged_row = dict(row)
        queue_row = queue_by_key.get(_event_key(row), {})
        merged_row["foot_x"] = queue_row.get("foot_x", "")
        merged_row["foot_y"] = queue_row.get("foot_y", "")
        merged_row["queue_event_id"] = queue_row.get("event_id", "")
        merged_row["evidence_buffer_json"] = queue_row.get("evidence_buffer_json", "")
        merged.append(merged_row)
    return merged


def build_identity_contingency_rows(matches):
    contingency = {}
    for row in matches:
        gt_id = str(row.get("gt_global_id", ""))
        pred_id = str(row.get("pred_identity_id", ""))
        if not gt_id or not pred_id:
            continue
        contingency.setdefault((gt_id, pred_id), 0)
        contingency[(gt_id, pred_id)] += 1
    rows = []
    for (gt_id, pred_id), count in sorted(contingency.items()):
        rows.append({"gt_identity_id": gt_id, "pred_identity_id": pred_id, "matched_event_count": count})
    return rows


def main(config_path: Path):
    eval_config = load_eval_config(config_path)
    project_root = resolve_path(config_path.parent, eval_config.get("project_root", "../../.."))
    benchmark_config_path = resolve_path(project_root, eval_config["offline_pipeline_config"])
    benchmark_offline_config = load_pipeline_config(benchmark_config_path)
    benchmark_output_root = resolve_path(project_root, eval_config.get("benchmark_output_root", benchmark_offline_config["output_root"]))
    evaluation_output_root = resolve_path(
        project_root,
        eval_config.get("evaluation_output_root", str(benchmark_output_root / "evaluation")),
    )
    gt_output_root = resolve_path(
        project_root,
        eval_config.get("gt_output_root", str(evaluation_output_root / "gt_reference")),
    )
    evaluation_output_root.mkdir(parents=True, exist_ok=True)

    gt_stage_inputs = build_gt_stage_inputs(project_root, benchmark_config_path, gt_output_root)

    gt_queue_rows = read_csv_rows(gt_stage_inputs["identity_queue_csv"])
    pred_track_rows = read_csv_rows(benchmark_output_root / "tracks" / "all_tracks_filtered.csv")
    pred_queue_rows = read_csv_rows(benchmark_output_root / "events" / "identity_resolution_queue.csv")
    resolved_rows = read_csv_rows(benchmark_output_root / "events" / "resolved_events.csv")
    resolved_event_rows = build_resolved_rows_for_event_matching(resolved_rows, pred_queue_rows)

    event_matches, unmatched_gt_events, unmatched_pred_events = match_event_rows_to_gt(
        gt_queue_rows,
        resolved_event_rows,
        frame_tolerance=safe_int(eval_config.get("event_match", {}).get("frame_tolerance", 3), 3),
        max_foot_distance_px=safe_float(eval_config.get("event_match", {}).get("max_foot_distance_px", 80.0), 80.0),
    )
    event_metrics = compute_event_level_idf1(gt_queue_rows, resolved_event_rows, event_matches)

    local_camera_id = eval_config.get("mota", {}).get("local_camera_id", "C1")
    pred_local_rows = [row for row in pred_track_rows if row.get("camera_id") == local_camera_id]
    gt_local_rows = build_local_tracking_gt_rows(project_root, benchmark_offline_config, pred_local_rows)
    local_tracking_metrics = compute_local_tracking_mota(
        gt_local_rows,
        pred_local_rows,
        iou_threshold=safe_float(eval_config.get("mota", {}).get("iou_threshold", 0.5), 0.5),
    )

    contingency_rows = build_identity_contingency_rows(event_matches)
    metrics_summary = {
        "evaluation_name": eval_config.get("evaluation_name", "single_source_sequential_c6_cache_benchmark"),
        "benchmark_output_root": str(benchmark_output_root),
        "gt_output_root": str(gt_output_root),
        "benchmark_config_path": str(benchmark_config_path),
        "gt_clip_definition": {
            "source_mode": "Cam6 sequential replay",
            "frame_range": {
                "start_frame": benchmark_offline_config.get("low_load", {}).get("start_frame", 0),
                "end_frame": benchmark_offline_config.get("low_load", {}).get("end_frame", ""),
                "frame_stride": benchmark_offline_config.get("low_load", {}).get("frame_stride", 1),
            },
            "actual_seconds": round(
                (safe_int(benchmark_offline_config.get("low_load", {}).get("end_frame", 0)) - safe_int(benchmark_offline_config.get("low_load", {}).get("start_frame", 0)))
                / max(59.94, 1e-9),
                3,
            ),
            "virtual_cameras": benchmark_offline_config.get("single_source_replay", {}).get("virtual_camera_ids", []),
        },
        "local_tracking_mota": local_tracking_metrics,
        "cross_camera_event_idf1": event_metrics,
        "event_match_summary": {
            "matched_event_count": len(event_matches),
            "unmatched_gt_event_count": len(unmatched_gt_events),
            "unmatched_pred_event_count": len(unmatched_pred_events),
        },
        "limitations": [
            "MOTA is measured on the first virtual replay camera only, because sequential replay duplicates one physical source video.",
            "IDF1 is measured on direction-filtered handoff events across C1->C4, not as a full public MTMCT benchmark over all pedestrians in all views.",
            "GT for this evaluation is generated from the same replay config with explicit gt_annotations mode, then matched back to inference outputs by camera/frame/foot-point proximity.",
        ],
    }

    save_json(evaluation_output_root / "quantitative_metrics_summary.json", metrics_summary)
    write_csv_rows(evaluation_output_root / "gt_entry_events.csv", gt_queue_rows)
    write_csv_rows(evaluation_output_root / "pred_resolved_events_for_eval.csv", resolved_event_rows)
    write_csv_rows(evaluation_output_root / "pred_event_gt_matches.csv", event_matches)
    write_csv_rows(
        evaluation_output_root / "pred_event_gt_unmatched_gt.csv",
        unmatched_gt_events,
        fieldnames=list(gt_queue_rows[0].keys()) if gt_queue_rows else [],
    )
    write_csv_rows(
        evaluation_output_root / "pred_event_gt_unmatched_pred.csv",
        unmatched_pred_events,
        fieldnames=list(resolved_event_rows[0].keys()) if resolved_event_rows else [],
    )
    write_csv_rows(evaluation_output_root / "identity_contingency.csv", contingency_rows)

    print(f"EVAL_OUTPUT_ROOT={evaluation_output_root}")
    print(f"IDF1={event_metrics['idf1']}")
    print(f"MOTA={local_tracking_metrics['mota']}")
    print(f"ID_SWITCHES={local_tracking_metrics['id_switches']}")
    print(f"FP={local_tracking_metrics['fp']}")
    print(f"FN={local_tracking_metrics['fn']}")
    return metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run quantitative MTMCT evaluation for the replay benchmark window.")
    parser.add_argument("--config", required=True, help="Path to quantitative evaluation YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main(Path(args.config).resolve())
