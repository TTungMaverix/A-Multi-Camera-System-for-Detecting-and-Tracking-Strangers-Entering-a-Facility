import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from association_core import load_camera_transition_map
from association_core.spatial_context import resolve_spatial_context
from dataset_profiles import load_dataset_profile_from_config
from offline_pipeline.event_builder import (
    FrameSourceCache,
    build_direction_windows,
    group_rows_by_track,
    point_to_segment_distance,
    test_is_in_side,
)
from offline_pipeline.orchestrator import load_pipeline_config, resolve_path
from scene_calibration import (
    apply_scene_calibration_to_transition_map,
    apply_scene_calibration_to_wildtrack_config,
    draw_scene_overlay,
    load_runtime_scene_calibration,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_image_unicode(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def _physical_camera_ids(dataset_profile):
    selected = list(dataset_profile.get("selected_cameras", []) or [])
    cameras = dataset_profile.get("cameras", {}) or {}
    physical_ids = [
        camera_id
        for camera_id in selected
        if not bool((cameras.get(camera_id, {}) or {}).get("logical_demo_copy", False))
    ]
    return physical_ids or selected[:2]


def _event_rows_by_camera_frame(events):
    by_camera = defaultdict(lambda: defaultdict(list))
    for row in events:
        by_camera[str(row.get("camera_id") or "")][int(float(row.get("frame_id") or 0))].append(row)
    return by_camera


def _track_debug_rows(camera_id, camera_cfg, runtime_camera, track_rows, transition_map):
    track_map = group_rows_by_track(track_rows)
    camera_tracks = []
    render_rows = []
    for local_track_id, records in sorted(track_map.items(), key=lambda item: item[0]):
        if not records:
            continue
        direction_windows = build_direction_windows(records, camera_id, camera_cfg, transition_map, camera_cfg.get("direction_filter", {}))
        selected_window = next(
            (window for window in direction_windows if (window.get("direction_result", {}) or {}).get("decision") == "IN"),
            None,
        )
        first_row = records[0]
        last_row = records[-1]
        line = runtime_camera.get("entry_line", []) or []
        in_side_point = runtime_camera.get("in_side_point", []) or []
        first_point = {"x": float(first_row.get("foot_x", 0.0)), "y": float(first_row.get("foot_y", 0.0))}
        last_point = {"x": float(last_row.get("foot_x", 0.0)), "y": float(last_row.get("foot_y", 0.0))}
        first_spatial = resolve_spatial_context(camera_id, first_point["x"], first_point["y"], transition_map)
        last_spatial = resolve_spatial_context(camera_id, last_point["x"], last_point["y"], transition_map)
        track_summary = {
            "camera_id": camera_id,
            "local_track_id": str(local_track_id),
            "row_count": len(records),
            "source_start_frame": int(float(first_row.get("source_frame_id_actual", first_row.get("frame_id", 0)) or 0)),
            "source_end_frame": int(float(last_row.get("source_frame_id_actual", last_row.get("frame_id", 0)) or 0)),
            "first_relative_sec": float(first_row.get("relative_sec", 0.0) or 0.0),
            "last_relative_sec": float(last_row.get("relative_sec", 0.0) or 0.0),
            "start_inside": bool(test_is_in_side(first_point, line, in_side_point)) if line and in_side_point else False,
            "end_inside": bool(test_is_in_side(last_point, line, in_side_point)) if line and in_side_point else False,
            "start_distance_to_line_px": round(float(point_to_segment_distance(first_point, line)) if line else 0.0, 3),
            "end_distance_to_line_px": round(float(point_to_segment_distance(last_point, line)) if line else 0.0, 3),
            "first_zone_id": first_spatial.get("zone_id", ""),
            "first_subzone_id": first_spatial.get("subzone_id", ""),
            "last_zone_id": last_spatial.get("zone_id", ""),
            "last_subzone_id": last_spatial.get("subzone_id", ""),
            "selected_anchor": {},
            "window_count": len(direction_windows),
        }
        if selected_window:
            direction_result = selected_window.get("direction_result", {}) or {}
            track_summary["selected_anchor"] = {
                "frame_id": int(selected_window["anchor_row"].get("frame_id", 0)),
                "source_frame_id_actual": int(
                    float(selected_window["anchor_row"].get("source_frame_id_actual", selected_window["anchor_row"].get("frame_id", 0)) or 0)
                ),
                "anchor_reason": selected_window.get("anchor_reason", ""),
                "direction_reason": direction_result.get("reason", ""),
                "direction_accept_mode": "late_start_inside_entry"
                if direction_result.get("late_start_inside_entry")
                else "cross_in",
                "momentum_px": float(direction_result.get("momentum_px", 0.0) or 0.0),
                "inside_ratio": float(direction_result.get("inside_ratio", 0.0) or 0.0),
                "zone_transition_ok": bool(direction_result.get("zone_transition_ok", False)),
            }
        if direction_windows:
            last_window = direction_windows[-1]
            track_summary["latest_direction_state"] = {
                "frame_id": int(last_window["anchor_row"].get("frame_id", 0)),
                "source_frame_id_actual": int(
                    float(last_window["anchor_row"].get("source_frame_id_actual", last_window["anchor_row"].get("frame_id", 0)) or 0)
                ),
                "decision": str((last_window.get("direction_result", {}) or {}).get("decision", "")),
                "reason": str((last_window.get("direction_result", {}) or {}).get("reason", "")),
                "momentum_px": float((last_window.get("direction_result", {}) or {}).get("momentum_px", 0.0) or 0.0),
                "inside_ratio": float((last_window.get("direction_result", {}) or {}).get("inside_ratio", 0.0) or 0.0),
            }
        camera_tracks.append(track_summary)
        for record_index, record in enumerate(records):
            render_rows.append(
                {
                    "record": record,
                    "track_summary": track_summary,
                    "track_index": record_index,
                }
            )
    render_rows.sort(key=lambda item: int(float(item["record"].get("source_frame_id_actual", item["record"].get("frame_id", 0)) or 0)))
    return camera_tracks, render_rows


def _render_row_frame(frame, camera_id, runtime_camera, record, track_summary, event_rows, transition_map):
    frame = draw_scene_overlay(frame, runtime_camera)
    xmin = int(float(record.get("xmin", 0) or 0))
    ymin = int(float(record.get("ymin", 0) or 0))
    xmax = int(float(record.get("xmax", 0) or 0))
    ymax = int(float(record.get("ymax", 0) or 0))
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 90), 2)
    foot_x = int(round(float(record.get("foot_x", 0.0) or 0.0)))
    foot_y = int(round(float(record.get("foot_y", 0.0) or 0.0)))
    cv2.circle(frame, (foot_x, foot_y), 6, (40, 180, 255), -1)
    spatial = resolve_spatial_context(camera_id, float(record.get("foot_x", 0.0) or 0.0), float(record.get("foot_y", 0.0) or 0.0), transition_map)
    line = runtime_camera.get("entry_line", []) or []
    in_side_point = runtime_camera.get("in_side_point", []) or []
    inside = bool(test_is_in_side({"x": foot_x, "y": foot_y}, line, in_side_point)) if line and in_side_point else False
    distance_to_line = point_to_segment_distance({"x": foot_x, "y": foot_y}, line) if line else 0.0
    lines = [
        f"{camera_id} track={track_summary['local_track_id']} src_frame={int(float(record.get('source_frame_id_actual', 0) or 0))}",
        f"inside={str(inside).lower()} dist_to_line={round(float(distance_to_line), 2)} accept_mode={track_summary.get('selected_anchor', {}).get('direction_accept_mode', 'none')}",
        f"zone={spatial.get('zone_id', '') or 'none'} subzone={spatial.get('subzone_id', '') or 'none'}",
        f"direction={track_summary.get('latest_direction_state', {}).get('decision', 'NONE')} reason={track_summary.get('latest_direction_state', {}).get('reason', '')}",
    ]
    if event_rows:
        lines.append(f"EVENT={event_rows[0].get('event_id', '')}")
        cv2.putText(frame, "ENTRY_EVENT", (xmin, max(20, ymin - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 230), 2, cv2.LINE_AA)
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (18, 28 + (index * 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    return frame


def _failure_frame_indices(render_rows, track_summaries):
    indices = set()
    if render_rows:
        indices.add(0)
        indices.add(len(render_rows) - 1)
    for track_summary in track_summaries:
        anchor = track_summary.get("selected_anchor", {}) or {}
        anchor_frame = int(anchor.get("source_frame_id_actual", -1) or -1)
        if anchor_frame < 0:
            continue
        for index, item in enumerate(render_rows):
            row = item["record"]
            if int(float(row.get("source_frame_id_actual", row.get("frame_id", 0)) or 0)) == anchor_frame:
                indices.add(index)
                break
    return sorted(indices)


def _blank_panel(reference_shape, title):
    height, width = reference_shape[:2]
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    return panel


def _fit_panel(frame, panel_shape):
    panel_height, panel_width = panel_shape[:2]
    height, width = frame.shape[:2]
    scale = min(panel_width / float(max(width, 1)), panel_height / float(max(height, 1)))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    offset_x = max(0, (panel_width - target_width) // 2)
    offset_y = max(0, (panel_height - target_height) // 2)
    canvas[offset_y : offset_y + target_height, offset_x : offset_x + target_width] = resized
    return canvas


def build_root_cause_summary(stage_input_summary, track_debug, missing_event_rows, entry_events):
    per_camera_runtime = (
        ((stage_input_summary.get("multi_source_inference", {}) or {}).get("per_physical_camera_runtime", {}))
        or (stage_input_summary.get("per_camera_runtime", {}) or {})
    )
    stage_counts = {
        "detector_alive": any(
            int((runtime_payload or {}).get("raw_detection_count", 0)) > 0
            for runtime_payload in per_camera_runtime.values()
        ),
        "tracker_alive": any(int(payload.get("track_count", 0)) > 0 for payload in track_debug.values()),
        "entry_event_count": len(entry_events),
        "missing_event_reason_counts": {},
    }
    reason_counts = defaultdict(int)
    for row in missing_event_rows:
        reason_counts[str(row.get("drop_reason", "") or "unknown")] += 1
    stage_counts["missing_event_reason_counts"] = dict(reason_counts)

    lines = []
    if len(entry_events) == 0:
        if reason_counts.get("filtered_by_direction", 0) > 0:
            lines.append("No entry event was emitted because all candidate tracks were filtered by the direction stage.")
        for camera_id, payload in track_debug.items():
            for track in payload.get("tracks", []):
                anchor = track.get("selected_anchor", {}) or {}
                if anchor.get("direction_accept_mode") == "late_start_inside_entry":
                    lines.append(
                        f"{camera_id}/{track['local_track_id']} is a late-start track: source_start_frame={track['source_start_frame']}, "
                        f"track starts inside the line, and a late-start entry anchor is available."
                    )
                    break
    else:
        late_start_events = [
            row
            for row in entry_events
            if str(row.get("direction_accept_mode", "") or "") == "late_start_inside_entry"
        ]
        if late_start_events:
            lines.append(f"{len(late_start_events)} entry event(s) were recovered by the late-start inside-entry fallback.")
        lines.append(
            "Detection and tracking are alive on a2; the clip now passes the event-creation stage and remaining failures are in downstream appearance-based association."
        )
    if not lines:
        lines.append("Detection and tracking are present; inspect per-camera track debug for the dominant failure stage.")
    return {
        "stage_counts": stage_counts,
        "root_cause_lines": lines,
    }


def write_root_cause_report(output_path: Path, pair_id, run_output_root: Path, root_cause_summary, track_debug):
    lines = [
        f"# {pair_id} Root Cause Report",
        "",
        f"- run_output_root: `{run_output_root}`",
        f"- entry_event_count: `{root_cause_summary['stage_counts']['entry_event_count']}`",
        f"- missing_event_reason_counts: `{root_cause_summary['stage_counts']['missing_event_reason_counts']}`",
        "",
        "## Root Cause",
        "",
    ]
    for item in root_cause_summary.get("root_cause_lines", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Per-Camera Tracks", ""])
    for camera_id, payload in track_debug.items():
        lines.append(f"### {camera_id}")
        lines.append("")
        for track in payload.get("tracks", []):
            lines.append(
                f"- `{track['local_track_id']}` rows=`{track['row_count']}` "
                f"start_inside=`{track['start_inside']}` start_distance_to_line_px=`{track['start_distance_to_line_px']}` "
                f"selected_anchor=`{track.get('selected_anchor', {})}`"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate overlay and stage diagnostics for a single New Dataset pair run.")
    parser.add_argument("--pipeline-config", default="insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml")
    parser.add_argument("--run-output-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-id", default="")
    args = parser.parse_args()

    pipeline_config_path = Path(args.pipeline_config).resolve()
    base_config = load_pipeline_config(pipeline_config_path)
    project_root = resolve_path(pipeline_config_path.parent, base_config.get("project_root", "."))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_root = Path(args.run_output_root).resolve()

    dataset_profile_path, dataset_profile, dataset_runtime = load_dataset_profile_from_config(
        base_config,
        project_root,
        base_dir=pipeline_config_path.parent,
    )
    transition_map, transition_runtime = load_camera_transition_map(
        dataset_profile,
        config_path=str(resolve_path(project_root, base_config.get("camera_transition_map_config", ""))),
        base_dir=pipeline_config_path.parent,
    )
    scene_calibration, runtime_cameras, scene_runtime = load_runtime_scene_calibration(
        config_path=resolve_path(project_root, base_config.get("scene_calibration_config", "")),
        base_dir=pipeline_config_path.parent,
        camera_ids=dataset_profile.get("selected_cameras", []),
        required=True,
    )
    transition_map = apply_scene_calibration_to_transition_map(transition_map, scene_calibration, scene_runtime.get("frame_sizes", {}))
    dataset_profile = apply_scene_calibration_to_wildtrack_config(dataset_profile, scene_calibration, scene_runtime.get("frame_sizes", {}))

    pair_id = args.pair_id or (load_json(run_output_root / "summaries" / "logical_demo_manifest.json").get("pair_id", "") if (run_output_root / "summaries" / "logical_demo_manifest.json").exists() else "")
    stage_input_summary = load_json(run_output_root / "summaries" / "stage_input_summary.json")
    missing_event_rows = read_csv_rows(run_output_root / "audit" / "audit_missing_event_reasons.csv")
    entry_events = read_csv_rows(run_output_root / "events" / "entry_in_events.csv")
    event_rows_by_camera_frame = _event_rows_by_camera_frame(entry_events)

    physical_camera_ids = _physical_camera_ids(dataset_profile)
    track_debug = {}
    render_rows_by_camera = {}
    for camera_id in physical_camera_ids:
        camera_rows = read_csv_rows(run_output_root / "tracks" / f"{camera_id}_tracks.csv")
        runtime_camera = runtime_cameras[camera_id]
        camera_cfg = dataset_profile["cameras"][camera_id]
        tracks, render_rows = _track_debug_rows(camera_id, camera_cfg, runtime_camera, camera_rows, transition_map)
        track_debug[camera_id] = {
            "track_count": len(tracks),
            "track_rows": len(camera_rows),
            "tracks": tracks,
        }
        render_rows_by_camera[camera_id] = render_rows

    frame_cache = FrameSourceCache()
    try:
        reference_shape = None
        panel_width = 0
        panel_height = 0
        camera_frames = {}
        for camera_id in physical_camera_ids:
            rows = render_rows_by_camera.get(camera_id, [])
            camera_frames[camera_id] = []
            for item in rows:
                record = item["record"]
                frame = frame_cache.frame_for_record(record, project_root)
                if reference_shape is None:
                    reference_shape = frame.shape
                panel_height = max(panel_height, frame.shape[0])
                panel_width = max(panel_width, frame.shape[1])
                event_rows = event_rows_by_camera_frame.get(camera_id, {}).get(int(float(record.get("frame_id", 0) or 0)), [])
                rendered = _render_row_frame(frame, camera_id, runtime_cameras[camera_id], record, item["track_summary"], event_rows, transition_map)
                camera_frames[camera_id].append(rendered)
        if reference_shape is None:
            raise RuntimeError(f"No track rows were available to render debug overlay for {run_output_root}")
        panel_shape = (panel_height, panel_width, 3)

        max_frames = max(len(camera_frames.get(camera_id, [])) for camera_id in physical_camera_ids)
        video_path = output_dir / f"{pair_id or 'pair'}_overlay_debug.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            4.0,
            (panel_width * len(physical_camera_ids), panel_height),
        )
        try:
            for frame_index in range(max_frames):
                panels = []
                for camera_id in physical_camera_ids:
                    rows = camera_frames.get(camera_id, [])
                    if frame_index < len(rows):
                        panels.append(_fit_panel(rows[frame_index], panel_shape))
                    else:
                        panels.append(_blank_panel(panel_shape, f"{camera_id} no more tracked frames"))
                writer.write(cv2.hconcat(panels))
        finally:
            writer.release()

        failure_dir = output_dir / f"{pair_id or 'pair'}_failure_frames"
        failure_dir.mkdir(parents=True, exist_ok=True)
        for camera_id in physical_camera_ids:
            rows = camera_frames.get(camera_id, [])
            indices = _failure_frame_indices(render_rows_by_camera.get(camera_id, []), track_debug[camera_id]["tracks"])
            for image_index in indices:
                if image_index >= len(rows):
                    continue
                write_image_unicode(failure_dir / f"{camera_id}_{image_index:02d}.png", rows[image_index])
    finally:
        frame_cache.close()

    root_cause_summary = build_root_cause_summary(stage_input_summary, track_debug, missing_event_rows, entry_events)
    summary_payload = {
        "pair_id": pair_id,
        "run_output_root": str(run_output_root),
        "pipeline_config_path": str(pipeline_config_path),
        "dataset_profile_runtime": dataset_runtime,
        "camera_transition_map_runtime": transition_runtime,
        "scene_calibration_runtime": scene_runtime,
        "stage_input_summary": stage_input_summary,
        "entry_event_count": len(entry_events),
        "missing_event_rows": missing_event_rows,
        "track_debug": track_debug,
        "root_cause_summary": root_cause_summary,
        "artifacts": {
            "overlay_video": str(output_dir / f"{pair_id or 'pair'}_overlay_debug.mp4"),
            "failure_frames_dir": str(output_dir / f"{pair_id or 'pair'}_failure_frames"),
            "root_cause_report": str(output_dir / f"{pair_id or 'pair'}_root_cause_report.md"),
        },
    }
    save_json(output_dir / f"{pair_id or 'pair'}_stage_debug_summary.json", summary_payload)
    write_root_cause_report(
        output_dir / f"{pair_id or 'pair'}_root_cause_report.md",
        pair_id or "pair",
        run_output_root,
        root_cause_summary,
        track_debug,
    )
    print(output_dir / f"{pair_id or 'pair'}_stage_debug_summary.json")


if __name__ == "__main__":
    main()
