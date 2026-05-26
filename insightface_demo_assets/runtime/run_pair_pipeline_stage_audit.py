import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from offline_pipeline.event_builder import build_direction_windows, point_to_segment_distance, test_point_in_polygon
from run_live_event_demo_server import (
    DEFAULT_CAMERA_SEQUENCE,
    _coerce_float,
    _coerce_int,
    _probe_video_metadata,
    _resolve_clip_video_sources,
    inspect_output_root,
    load_association_decisions,
    load_association_summary,
    load_face_body_usage_summary,
    load_face_resolution_summary,
    load_identity_timeline,
    load_known_db_summary,
    load_latest_events,
    load_offline_pipeline_summary,
    load_reid_handoffs,
    load_resolved_event_rows,
)
from scene_calibration import build_runtime_camera_calibration, draw_scene_overlay, load_scene_calibration, probe_frame_from_source
from offline_pipeline.direction_logic import is_in_side


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit one pair through calibration, tracking, direction, and artifact stages.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene-calibration-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--known-db-root", required=True)
    parser.add_argument("--audit-output-dir", required=True)
    return parser.parse_args()


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += float(point[0]) * float(next_point[1])
        total -= float(next_point[0]) * float(point[1])
    return abs(total) / 2.0


def summarize_number_series(values):
    numeric = [float(item) for item in values if item not in ("", None)]
    if not numeric:
        return {"min": "NOT_MEASURED", "median": "NOT_MEASURED", "max": "NOT_MEASURED"}
    ordered = sorted(numeric)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "min": round(min(ordered), 4),
        "median": round(median, 4),
        "max": round(max(ordered), 4),
    }


def annotate_frame(frame, lines):
    canvas = frame.copy()
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            str(line),
            (14, 24 + (index * 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(line),
            (14, 24 + (index * 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (18, 18, 18),
            1,
            cv2.LINE_AA,
        )
    return canvas


def truthy_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def camera_output_prefix(pair_id: str, camera_id: str, suffix: str):
    return f"{pair_id}_{camera_id}_{suffix}"


def summarize_video_sources(pair_id: str, dataset_root: Path):
    sources = _resolve_clip_video_sources(pair_id, dataset_root)
    summary = {}
    for camera_id in DEFAULT_CAMERA_SEQUENCE:
        source = sources.get(camera_id, {})
        path = source.get("path")
        metadata = _probe_video_metadata(path)
        summary[camera_id] = {
            "path": str(path) if path else "",
            "source_note": source.get("source_note", "MISSING"),
            "exists": bool(path and Path(path).exists()),
            "width": metadata.get("width", 0),
            "height": metadata.get("height", 0),
            "fps": metadata.get("fps", 0.0),
            "frame_count": metadata.get("frame_count", 0),
            "duration_sec": metadata.get("duration_sec", 0.0),
            "first_frame_readable": bool(metadata.get("opened", False)),
        }
    return summary


def summarize_calibration(calibration_path: Path, pair_id: str, video_summary):
    calibration, runtime_info = load_scene_calibration(
        config_path=str(calibration_path),
        base_dir=calibration_path.parent,
        required=False,
        camera_ids=list(DEFAULT_CAMERA_SEQUENCE),
    )
    cameras = calibration.get("cameras", {}) or {}
    summary = {
        "config_path": str(calibration_path),
        "config_exists": calibration_path.exists(),
        "config_load_ok": calibration_path.exists() and not runtime_info.get("errors"),
        "runtime_errors": list(runtime_info.get("errors", [])),
        "runtime_warnings": list(runtime_info.get("warnings", [])),
        "cameras": {},
        "frame_size_matches": True,
    }
    runtime_cameras = {}
    for camera_id in DEFAULT_CAMERA_SEQUENCE:
        camera_cfg = cameras.get(camera_id, {}) or {}
        video_meta = video_summary.get(camera_id, {})
        frame_width = _coerce_int(video_meta.get("width"), _coerce_int((camera_cfg.get("frame_size_ref", {}) or {}).get("width"), 0))
        frame_height = _coerce_int(video_meta.get("height"), _coerce_int((camera_cfg.get("frame_size_ref", {}) or {}).get("height"), 0))
        runtime_camera = build_runtime_camera_calibration(camera_cfg, frame_width or 1, frame_height or 1)
        runtime_cameras[camera_id] = runtime_camera
        roi_polygon_norm = ((camera_cfg.get("processing_roi", {}) or {}).get("polygon") or [])
        entry_line_norm = ((camera_cfg.get("entry_line", {}) or {}).get("points") or [])
        in_side_point_norm = ((camera_cfg.get("entry_line", {}) or {}).get("in_side_point") or [])
        frame_ref = camera_cfg.get("frame_size_ref", {}) or {}
        width_match = _coerce_int(frame_ref.get("width"), 0) == _coerce_int(video_meta.get("width"), 0)
        height_match = _coerce_int(frame_ref.get("height"), 0) == _coerce_int(video_meta.get("height"), 0)
        if not (width_match and height_match):
            summary["frame_size_matches"] = False
        roi_area_ratio = 0.0
        if frame_width > 0 and frame_height > 0:
            roi_area_ratio = round(
                polygon_area(runtime_camera.get("processing_roi", [])) / float(frame_width * frame_height),
                6,
            )
        warnings = []
        if roi_area_ratio < 0.02:
            warnings.append("roi_area_tiny")
        if roi_area_ratio > 0.95:
            warnings.append("roi_area_huge")
        if len(runtime_camera.get("entry_line", [])) == 2:
            for point in runtime_camera.get("entry_line", []):
                if point[0] < 0 or point[0] > frame_width or point[1] < 0 or point[1] > frame_height:
                    warnings.append("entry_line_out_of_frame")
                    break
        summary["cameras"][camera_id] = {
            "present": camera_id in cameras,
            "role": camera_cfg.get("role", ""),
            "frame_size_ref": frame_ref,
            "frame_size_ref_matches_video": bool(width_match and height_match),
            "roi_point_count": len(roi_polygon_norm),
            "roi_area_ratio": roi_area_ratio,
            "entry_line_point_count": len(entry_line_norm),
            "entry_line_points_norm": entry_line_norm,
            "entry_line_points_px": runtime_camera.get("entry_line", []),
            "in_side_point_present": len(in_side_point_norm) == 2,
            "in_side_point_norm": in_side_point_norm,
            "in_side_point_px": runtime_camera.get("in_side_point", []),
            "anchor_point_mode": camera_cfg.get("anchor_point_mode", ""),
            "warnings": warnings,
        }
    return summary, calibration, runtime_cameras


def load_transition_map(output_root: Path):
    candidates = [
        output_root / "association_logs" / "camera_transition_map_runtime.json",
        output_root / "runtime" / "association_logs" / "camera_transition_map_runtime.json",
    ]
    for candidate in candidates:
        payload = load_json(candidate)
        if isinstance(payload, dict):
            return payload
    return {}


def inspect_artifact_payloads(project_root: Path, output_root: Path):
    status = inspect_output_root(output_root)
    latest_events = load_latest_events(output_root) if status.get("artifacts_found") else []
    resolved_rows = load_resolved_event_rows(output_root) if status.get("artifacts_found") else {}
    timeline_rows = load_identity_timeline(output_root) if status.get("artifacts_found") else []
    handoffs = load_reid_handoffs(output_root) if status.get("artifacts_found") else []
    association_decisions = load_association_decisions(output_root) if status.get("artifacts_found") else []
    face_resolution_summary = load_face_resolution_summary(output_root) if status.get("artifacts_found") else {}
    face_body_usage_summary = load_face_body_usage_summary(output_root) if status.get("artifacts_found") else {}
    association_summary = load_association_summary(output_root) if status.get("artifacts_found") else {}
    offline_pipeline_summary = load_offline_pipeline_summary(output_root) if status.get("artifacts_found") else {}
    known_db_summary = load_known_db_summary(project_root)
    track_rows_by_camera = {}
    for camera_id in DEFAULT_CAMERA_SEQUENCE:
        track_rows_by_camera[camera_id] = load_csv_rows(output_root / "tracks" / f"{camera_id}_tracks.csv")
    all_tracks_filtered = load_csv_rows(output_root / "tracks" / "all_tracks_filtered.csv")
    missing_reason_rows = load_csv_rows(output_root / "audit" / "audit_missing_event_reasons.csv")
    assignment_audit_rows = load_csv_rows(output_root / "audit" / "entry_event_assignment_audit.csv")
    face_metrics = face_body_usage_summary.get("metrics", face_body_usage_summary) if isinstance(face_body_usage_summary, dict) else {}
    face_resolution_mode_b = face_resolution_summary.get("mode_b_true_assoc", {}) if isinstance(face_resolution_summary, dict) else {}
    known_runtime = face_resolution_summary.get("known_db_runtime", {}) if isinstance(face_resolution_summary, dict) else {}
    artifact_summary = {
        "status": status,
        "latest_events_count": len(latest_events),
        "resolved_events_count": len(resolved_rows),
        "timeline_identity_count": len(timeline_rows),
        "handoff_count": len(handoffs),
        "association_decision_count": len(association_decisions),
        "track_row_count_per_camera": {camera_id: len(rows) for camera_id, rows in track_rows_by_camera.items()},
        "all_tracks_filtered_count": len(all_tracks_filtered),
        "missing_event_reason_row_count": len(missing_reason_rows),
        "entry_assignment_audit_row_count": len(assignment_audit_rows),
        "face_candidate_count": face_metrics.get("face_candidate_count", "NOT_MEASURED"),
        "face_embedding_created_count": face_metrics.get("face_embedding_created_count", "NOT_MEASURED"),
        "known_match_success_count": face_metrics.get("known_face_match_success_count", "NOT_MEASURED"),
        "known_db_identity_count": known_runtime.get("identities_loaded", known_db_summary.get("identity_count", 0)),
        "known_db_embedding_count": known_runtime.get("embedding_count", known_db_summary.get("embedding_count", 0)),
        "offline_pipeline_total_sec": (offline_pipeline_summary.get("timings_sec", {}) or {}).get("total_pipeline_sec", "NOT_MEASURED")
        if isinstance(offline_pipeline_summary, dict)
        else "NOT_MEASURED",
        "mode_b_known_event_count": face_resolution_mode_b.get("known_event_count", "NOT_MEASURED"),
        "mode_b_unknown_event_count": face_resolution_mode_b.get("unknown_event_count", "NOT_MEASURED"),
    }
    return {
        "artifact_summary": artifact_summary,
        "latest_events": latest_events,
        "resolved_rows": resolved_rows,
        "timeline_rows": timeline_rows,
        "handoffs": handoffs,
        "association_decisions": association_decisions,
        "face_resolution_summary": face_resolution_summary,
        "face_body_usage_summary": face_body_usage_summary,
        "association_summary": association_summary,
        "offline_pipeline_summary": offline_pipeline_summary,
        "track_rows_by_camera": track_rows_by_camera,
        "all_tracks_filtered": all_tracks_filtered,
        "missing_reason_rows": missing_reason_rows,
        "assignment_audit_rows": assignment_audit_rows,
    }


def summarize_missing_reason_rows(rows):
    if not rows:
        return {}
    counts = Counter()
    for row in rows:
        reason = row.get("reason") or row.get("reason_code") or row.get("failure_stage") or "UNKNOWN"
        count_value = _coerce_int(row.get("count"), 1)
        counts[str(reason)] += max(1, count_value)
    return dict(counts)


def summarize_tracks(track_rows_by_camera, runtime_cameras, direction_cfg, transition_map):
    per_camera = {}
    debug_rows = []
    roi_debug_rows = []
    track_record_map = {}
    total_track_count = 0
    direction_candidate_track_count = 0
    direction_candidate_window_count = 0
    entry_in_track_count = 0
    out_track_count = 0
    rejected_direction_track_count = 0
    reject_reason_counts = Counter()
    line_crossing_track_count = 0
    late_start_track_count = 0
    detection_count_per_camera = {}
    roi_inside_count_per_camera = {}
    roi_outside_count_per_camera = {}
    for camera_id, rows in track_rows_by_camera.items():
        detection_count_per_camera[camera_id] = "NOT_MEASURED"
        runtime_camera = runtime_cameras.get(camera_id, {})
        roi_polygon = runtime_camera.get("processing_roi", []) or []
        entry_line = runtime_camera.get("entry_line", []) or []
        in_side_point = runtime_camera.get("in_side_point", []) or []
        grouped = defaultdict(list)
        inside_rows = 0
        outside_rows = 0
        for row in rows:
            grouped[str(row.get("local_track_id") or row.get("track_id") or "UNKNOWN")].append(row)
            foot_x = _coerce_float(row.get("foot_x"), 0.0)
            foot_y = _coerce_float(row.get("foot_y"), 0.0)
            roi_inside = truthy_bool(row.get("roi_footpoint_inside")) if "roi_footpoint_inside" in row else (
                test_point_in_polygon(foot_x, foot_y, roi_polygon) if roi_polygon else False
            )
            inside_rows += 1 if roi_inside else 0
            outside_rows += 0 if roi_inside else 1
            line_distance = point_to_segment_distance({"x": foot_x, "y": foot_y}, entry_line) if len(entry_line) == 2 else "NOT_MEASURED"
            point_inside_line = is_in_side({"x": foot_x, "y": foot_y}, entry_line, in_side_point) if len(entry_line) == 2 and len(in_side_point) == 2 else "NOT_MEASURED"
            roi_debug_rows.append(
                {
                    "camera_id": camera_id,
                    "local_track_id": row.get("local_track_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "source_frame_id_actual": row.get("source_frame_id_actual", ""),
                    "relative_sec": row.get("relative_sec", ""),
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                    "roi_footpoint_inside": roi_inside,
                    "line_distance_px": line_distance,
                    "in_side_flag": point_inside_line,
                    "pair_id": row.get("pair_id", ""),
                }
            )
        roi_inside_count_per_camera[camera_id] = inside_rows
        roi_outside_count_per_camera[camera_id] = outside_rows
        camera_track_rows = []
        for local_track_id, track_rows in grouped.items():
            ordered_rows = sorted(
                track_rows,
                key=lambda item: (
                    _coerce_int(item.get("source_frame_id_actual"), _coerce_int(item.get("frame_id"), 0)),
                    _coerce_int(item.get("frame_id"), 0),
                ),
            )
            windows = build_direction_windows(ordered_rows, camera_id, runtime_camera, transition_map, direction_cfg)
            direction_candidate_track_count += 1 if windows else 0
            direction_candidate_window_count += len(windows)
            any_in = any((window.get("direction_result", {}) or {}).get("decision") == "IN" for window in windows)
            any_out = any((window.get("direction_result", {}) or {}).get("decision") == "OUT" for window in windows)
            any_cross = any(
                bool((window.get("direction_result", {}) or {}).get("cross_in"))
                or bool((window.get("direction_result", {}) or {}).get("cross_out"))
                for window in windows
            )
            any_late_start = any(bool((window.get("direction_result", {}) or {}).get("late_start_inside_entry")) for window in windows)
            final_window = next((window for window in windows if (window.get("direction_result", {}) or {}).get("decision") == "IN"), None)
            if final_window is None and windows:
                final_window = windows[-1]
            final_direction = ((final_window or {}).get("direction_result", {}) or {}).get("decision", "NONE")
            final_reason = ((final_window or {}).get("direction_result", {}) or {}).get("reason", "NO_DIRECTION_WINDOWS")
            if any_in:
                entry_in_track_count += 1
            elif any_out:
                out_track_count += 1
            elif windows:
                rejected_direction_track_count += 1
                reject_reason_counts[str(final_reason)] += 1
            if any_cross:
                line_crossing_track_count += 1
            if any_late_start:
                late_start_track_count += 1
            min_distance = "NOT_MEASURED"
            if len(entry_line) == 2:
                min_distance = round(
                    min(point_to_segment_distance({"x": _coerce_float(row.get("foot_x"), 0.0), "y": _coerce_float(row.get("foot_y"), 0.0)}, entry_line) for row in ordered_rows),
                    3,
                )
            starts_inside_roi = truthy_bool(ordered_rows[0].get("roi_footpoint_inside")) if "roi_footpoint_inside" in ordered_rows[0] else (
                test_point_in_polygon(_coerce_float(ordered_rows[0].get("foot_x"), 0.0), _coerce_float(ordered_rows[0].get("foot_y"), 0.0), roi_polygon)
                if roi_polygon
                else False
            )
            any_inside_roi = any(truthy_bool(row.get("roi_footpoint_inside")) for row in ordered_rows) if "roi_footpoint_inside" in ordered_rows[0] else any(
                test_point_in_polygon(_coerce_float(row.get("foot_x"), 0.0), _coerce_float(row.get("foot_y"), 0.0), roi_polygon)
                for row in ordered_rows
            )
            debug_row = {
                "camera_id": camera_id,
                "local_track_id": local_track_id,
                "track_row_count": len(ordered_rows),
                "first_frame_id": _coerce_int(ordered_rows[0].get("frame_id"), 0),
                "last_frame_id": _coerce_int(ordered_rows[-1].get("frame_id"), 0),
                "first_source_frame_id_actual": _coerce_int(ordered_rows[0].get("source_frame_id_actual"), 0),
                "last_source_frame_id_actual": _coerce_int(ordered_rows[-1].get("source_frame_id_actual"), 0),
                "first_relative_sec": round(_coerce_float(ordered_rows[0].get("relative_sec"), 0.0), 3),
                "last_relative_sec": round(_coerce_float(ordered_rows[-1].get("relative_sec"), 0.0), 3),
                "starts_inside_roi": starts_inside_roi,
                "any_inside_roi": any_inside_roi,
                "line_crossing_detected": any_cross,
                "late_start_inside_entry": any_late_start,
                "direction_window_count": len(windows),
                "direction_decision": final_direction,
                "direction_reason": final_reason,
                "min_distance_to_entry_line_px": min_distance,
            }
            camera_track_rows.append(debug_row)
            debug_rows.append(debug_row)
            track_record_map[(camera_id, local_track_id)] = {
                "rows": ordered_rows,
                "runtime_camera": runtime_camera,
            }
        total_track_count += len(camera_track_rows)
        per_camera[camera_id] = {
            "track_count": len(camera_track_rows),
            "track_ids": [row["local_track_id"] for row in camera_track_rows],
            "roi_inside_row_count": inside_rows,
            "roi_outside_row_count": outside_rows,
            "direction_candidate_track_count": sum(1 for row in camera_track_rows if row["direction_window_count"] > 0),
            "entry_in_track_count": sum(1 for row in camera_track_rows if row["direction_decision"] == "IN"),
            "out_track_count": sum(1 for row in camera_track_rows if row["direction_decision"] == "OUT"),
            "rejected_direction_track_count": sum(
                1
                for row in camera_track_rows
                if row["direction_window_count"] > 0 and row["direction_decision"] not in {"IN", "OUT"}
            ),
        }
    summary = {
        "detection_count_per_camera": detection_count_per_camera,
        "detection_audit_status": "DETECTION_AUDIT_NOT_AVAILABLE",
        "roi_inside_count_per_camera": roi_inside_count_per_camera,
        "roi_outside_count_per_camera": roi_outside_count_per_camera,
        "track_count_per_camera": {camera_id: data["track_count"] for camera_id, data in per_camera.items()},
        "total_track_count": total_track_count,
        "direction_candidate_track_count": direction_candidate_track_count,
        "direction_candidate_window_count": direction_candidate_window_count,
        "entry_in_track_count": entry_in_track_count,
        "out_track_count": out_track_count,
        "rejected_direction_track_count": rejected_direction_track_count,
        "reject_reason_counts": dict(reject_reason_counts),
        "line_crossing_track_count": line_crossing_track_count,
        "late_start_track_count": late_start_track_count,
        "per_camera": per_camera,
    }
    return summary, debug_rows, roi_debug_rows, track_record_map


def draw_track_debug_overlay(frame, runtime_camera, track_rows, pair_id: str, camera_id: str, label: str):
    canvas = draw_scene_overlay(frame.copy(), runtime_camera)
    points = []
    for row in track_rows:
        x = int(round(_coerce_float(row.get("foot_x"), 0.0)))
        y = int(round(_coerce_float(row.get("foot_y"), 0.0)))
        points.append((x, y))
        cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1)
    for index in range(1, len(points)):
        cv2.line(canvas, points[index - 1], points[index], (0, 255, 255), 2, cv2.LINE_AA)
    if track_rows:
        row = track_rows[-1]
        xmin = _coerce_int(row.get("xmin"), 0)
        ymin = _coerce_int(row.get("ymin"), 0)
        xmax = _coerce_int(row.get("xmax"), 0)
        ymax = _coerce_int(row.get("ymax"), 0)
        if xmax > xmin and ymax > ymin:
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 200, 80), 2)
    return annotate_frame(
        canvas,
        [
            f"PAIR {pair_id} | CAM {camera_id}",
            label,
            f"TRACK_ROWS {len(track_rows)}",
        ],
    )


def generate_overlays(pair_id: str, video_summary, runtime_cameras, track_record_map, debug_rows, output_dir: Path):
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for camera_id in DEFAULT_CAMERA_SEQUENCE:
        video_path = video_summary.get(camera_id, {}).get("path", "")
        runtime_camera = runtime_cameras.get(camera_id, {})
        if not video_path:
            continue
        frame = probe_frame_from_source("file", video_path, frame_idx=0)
        overlay = draw_scene_overlay(frame.copy(), runtime_camera)
        overlay = annotate_frame(
            overlay,
            [
                f"PAIR {pair_id} | CAM {camera_id}",
                f"SRC {Path(video_path).name}",
                "FRAME 0 | calibration geometry",
            ],
        )
        output_path = overlay_dir / f"{camera_output_prefix(pair_id, camera_id, 'calibration_overlay')}.png"
        cv2.imwrite(str(output_path), overlay)
        generated.append(str(output_path))
    for debug_row in debug_rows:
        camera_id = debug_row["camera_id"]
        key = (camera_id, debug_row["local_track_id"])
        payload = track_record_map.get(key)
        if not payload:
            continue
        track_rows = payload["rows"]
        runtime_camera = payload["runtime_camera"]
        source_row = track_rows[0]
        video_path = source_row.get("video_path") or video_summary.get(camera_id, {}).get("path", "")
        if not video_path:
            continue
        first_frame_idx = _coerce_int(source_row.get("source_frame_id_actual"), _coerce_int(source_row.get("frame_id"), 0))
        first_frame = probe_frame_from_source("file", video_path, frame_idx=first_frame_idx)
        first_overlay = draw_track_debug_overlay(
            first_frame,
            runtime_camera,
            track_rows,
            pair_id,
            camera_id,
            f"TRACK {debug_row['local_track_id']} | FIRST_FRAME {first_frame_idx}",
        )
        first_suffix = f"{debug_row['local_track_id']}_first_track"
        first_output = overlay_dir / f"{camera_output_prefix(pair_id, camera_id, first_suffix)}.png"
        cv2.imwrite(str(first_output), first_overlay)
        generated.append(str(first_output))
        if debug_row["min_distance_to_entry_line_px"] != "NOT_MEASURED":
            nearest_row = min(
                track_rows,
                key=lambda row: point_to_segment_distance(
                    {"x": _coerce_float(row.get("foot_x"), 0.0), "y": _coerce_float(row.get("foot_y"), 0.0)},
                    runtime_camera.get("entry_line", []),
                ),
            )
            nearest_idx = _coerce_int(nearest_row.get("source_frame_id_actual"), _coerce_int(nearest_row.get("frame_id"), 0))
            nearest_frame = probe_frame_from_source("file", video_path, frame_idx=nearest_idx)
            nearest_overlay = draw_track_debug_overlay(
                nearest_frame,
                runtime_camera,
                track_rows,
                pair_id,
                camera_id,
                f"TRACK {debug_row['local_track_id']} | NEAREST_LINE {nearest_idx}",
            )
            nearest_suffix = f"{debug_row['local_track_id']}_nearest_line"
            nearest_output = overlay_dir / f"{camera_output_prefix(pair_id, camera_id, nearest_suffix)}.png"
            cv2.imwrite(str(nearest_output), nearest_overlay)
            generated.append(str(nearest_output))
    return generated


def classify_primary_fail_stage(video_summary, calibration_summary, artifact_bundle, track_summary):
    if any(not payload.get("exists") or not payload.get("first_frame_readable") for payload in video_summary.values()):
        return "VIDEO_SOURCE_MISSING"
    if not calibration_summary.get("config_exists") or not calibration_summary.get("config_load_ok") or not calibration_summary.get("frame_size_matches"):
        return "CALIBRATION_INVALID"
    if not artifact_bundle["artifact_summary"]["status"].get("exists"):
        return "OUTPUT_ROOT_MISSING"
    total_tracks = track_summary.get("total_track_count", 0)
    event_count = artifact_bundle["artifact_summary"]["latest_events_count"]
    resolved_count = artifact_bundle["artifact_summary"]["resolved_events_count"]
    if total_tracks <= 0:
        return "TRACKING_ZERO"
    if event_count > 0:
        return "UNKNOWN"
    inside_rows = sum(track_summary.get("roi_inside_count_per_camera", {}).values())
    outside_rows = sum(track_summary.get("roi_outside_count_per_camera", {}).values())
    if inside_rows <= 0 and outside_rows > 0:
        return "DETECTIONS_OUTSIDE_ROI"
    if track_summary.get("line_crossing_track_count", 0) <= 0 and track_summary.get("late_start_track_count", 0) <= 0:
        return "TRACKS_DO_NOT_CROSS_ENTRY_LINE"
    if track_summary.get("entry_in_track_count", 0) <= 0 and track_summary.get("direction_candidate_track_count", 0) > 0:
        return "DIRECTION_FILTER_REJECTED_ALL"
    if resolved_count > 0 or track_summary.get("entry_in_track_count", 0) > 0:
        return "EVENT_BUILDER_EMPTY"
    return "UNKNOWN"


def build_user_explanation(pair_id: str, primary_fail_stage: str):
    if primary_fail_stage == "OUTPUT_ROOT_MISSING":
        return f"No pipeline artifacts were found for {pair_id}. Run artifact generation first."
    if primary_fail_stage == "VIDEO_SOURCE_MISSING":
        return f"Video source resolution failed for {pair_id}. Check dataset paths before trusting demo playback."
    if primary_fail_stage == "CALIBRATION_INVALID":
        return f"Calibration is invalid for {pair_id}. Fix manual ROI or entry-line config before trusting direction output."
    if primary_fail_stage == "TRACKING_ZERO":
        return f"No tracks were materialized for {pair_id}. Detection or tracking produced zero usable rows."
    if primary_fail_stage == "DETECTIONS_OUTSIDE_ROI":
        return f"Detections or tracks exist for {pair_id}, but they stay outside the calibrated ROI."
    if primary_fail_stage == "TRACKS_DO_NOT_CROSS_ENTRY_LINE":
        return f"Tracks exist for {pair_id}, but they do not cross or start near the entry line strongly enough to trigger ENTRY_IN."
    if primary_fail_stage == "DIRECTION_FILTER_REJECTED_ALL":
        return f"Direction filtering rejected all candidate tracks for {pair_id}. Check ROI, entry line, and IN-side geometry."
    if primary_fail_stage == "EVENT_BUILDER_EMPTY":
        return f"Direction candidates exist for {pair_id}, but no ENTRY_IN events were materialized into the final artifacts."
    return f"No ENTRY_IN events were materialized for {pair_id}. Run the stage audit and inspect overlays before changing thresholds."


def main():
    args = parse_args()
    project_root = Path.cwd().resolve()
    pair_id = args.pair_id.strip()
    dataset_root = Path(args.dataset_root).resolve()
    calibration_path = (project_root / args.scene_calibration_config).resolve() if not Path(args.scene_calibration_config).is_absolute() else Path(args.scene_calibration_config).resolve()
    output_root = (project_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()
    known_db_root = Path(args.known_db_root).resolve()
    audit_output_dir = (project_root / args.audit_output_dir).resolve() if not Path(args.audit_output_dir).is_absolute() else Path(args.audit_output_dir).resolve()

    video_summary = summarize_video_sources(pair_id, dataset_root)
    calibration_summary, calibration_payload, runtime_cameras = summarize_calibration(calibration_path, pair_id, video_summary)
    artifact_bundle = inspect_artifact_payloads(project_root, output_root)
    direction_cfg = (calibration_payload.get("direction_filter", {}) or {})
    transition_map = load_transition_map(output_root)
    track_summary, debug_rows, roi_debug_rows, track_record_map = summarize_tracks(
        artifact_bundle["track_rows_by_camera"],
        runtime_cameras,
        direction_cfg,
        transition_map,
    )
    missing_reason_counts = summarize_missing_reason_rows(artifact_bundle["missing_reason_rows"])
    primary_fail_stage = classify_primary_fail_stage(video_summary, calibration_summary, artifact_bundle, track_summary)
    explanation = build_user_explanation(pair_id, primary_fail_stage)

    overlays = generate_overlays(pair_id, video_summary, runtime_cameras, track_record_map, debug_rows, audit_output_dir)

    tracks_debug_path = audit_output_dir / f"{pair_id}_tracks_debug.csv"
    detection_roi_debug_path = audit_output_dir / f"{pair_id}_detection_roi_debug.csv"
    write_csv(
        tracks_debug_path,
        debug_rows,
        [
            "camera_id",
            "local_track_id",
            "track_row_count",
            "first_frame_id",
            "last_frame_id",
            "first_source_frame_id_actual",
            "last_source_frame_id_actual",
            "first_relative_sec",
            "last_relative_sec",
            "starts_inside_roi",
            "any_inside_roi",
            "line_crossing_detected",
            "late_start_inside_entry",
            "direction_window_count",
            "direction_decision",
            "direction_reason",
            "min_distance_to_entry_line_px",
        ],
    )
    write_csv(
        detection_roi_debug_path,
        roi_debug_rows,
        [
            "camera_id",
            "local_track_id",
            "frame_id",
            "source_frame_id_actual",
            "relative_sec",
            "foot_x",
            "foot_y",
            "roi_footpoint_inside",
            "line_distance_px",
            "in_side_flag",
            "pair_id",
        ],
    )

    face_metrics = artifact_bundle["artifact_summary"]
    summary = {
        "pair_id": pair_id,
        "dataset_root": str(dataset_root),
        "scene_calibration_config": str(calibration_path),
        "output_root": str(output_root),
        "known_db_root": str(known_db_root),
        "video_source": {
            "pair_id": pair_id,
            "cameras": video_summary,
        },
        "calibration": calibration_summary,
        "artifacts": artifact_bundle["artifact_summary"],
        "tracking": track_summary,
        "direction_event_diagnosis": {
            "direction_candidate_track_count": track_summary.get("direction_candidate_track_count", 0),
            "direction_candidate_window_count": track_summary.get("direction_candidate_window_count", 0),
            "entry_in_track_count": track_summary.get("entry_in_track_count", 0),
            "out_track_count": track_summary.get("out_track_count", 0),
            "rejected_direction_track_count": track_summary.get("rejected_direction_track_count", 0),
            "reject_reason_counts": track_summary.get("reject_reason_counts", {}),
            "line_crossing_track_count": track_summary.get("line_crossing_track_count", 0),
            "late_start_track_count": track_summary.get("late_start_track_count", 0),
            "audit_missing_event_reason_counts": missing_reason_counts,
            "latest_event_count": artifact_bundle["artifact_summary"]["latest_events_count"],
            "resolved_event_count": artifact_bundle["artifact_summary"]["resolved_events_count"],
        },
        "face_known": {
            "face_candidate_count": face_metrics.get("face_candidate_count", "NOT_MEASURED"),
            "face_embedding_created_count": face_metrics.get("face_embedding_created_count", "NOT_MEASURED"),
            "known_match_success_count": face_metrics.get("known_match_success_count", "NOT_MEASURED"),
            "known_db_identity_count": face_metrics.get("known_db_identity_count", "NOT_MEASURED"),
            "known_db_embedding_count": face_metrics.get("known_db_embedding_count", "NOT_MEASURED"),
        },
        "primary_fail_stage": primary_fail_stage,
        "user_explanation": explanation,
        "overlay_paths": overlays,
        "tracks_debug_csv": str(tracks_debug_path),
        "detection_roi_debug_csv": str(detection_roi_debug_path),
    }
    summary_path = audit_output_dir / f"{pair_id}_stage_audit_summary.json"
    save_json(summary_path, summary)
    print(f"PAIR_ID={pair_id}")
    print(f"AUDIT_SUMMARY={summary_path}")
    print(f"PRIMARY_FAIL_STAGE={primary_fail_stage}")
    print(f"ENTRY_IN_COUNT={artifact_bundle['artifact_summary']['latest_events_count']}")
    print(f"TRACK_COUNT={track_summary.get('total_track_count', 0)}")
    print(f"KNOWN_DB={face_metrics.get('known_db_identity_count', 'NOT_MEASURED')}/{face_metrics.get('known_db_embedding_count', 'NOT_MEASURED')}")


if __name__ == "__main__":
    main()
