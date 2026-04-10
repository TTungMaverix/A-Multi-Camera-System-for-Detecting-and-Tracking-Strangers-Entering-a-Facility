import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from association_core.spatial_context import resolve_spatial_context
from offline_pipeline.direction_logic import evaluate_direction


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fieldnames_for_rows(rows, preferred=None):
    ordered = list(preferred or [])
    seen = set(ordered)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def test_point_in_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        dy = yj - yi
        if abs(dy) < 1e-9:
            dy = 1e-9
        intersect = ((yi > y) != (yj > y)) and (x < ((xj - xi) * (y - yi) / dy + xi))
        if intersect:
            inside = not inside
        j = i
    return inside


def get_line_cross_value(point, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    return ((x2 - x1) * (point["y"] - y1)) - ((y2 - y1) * (point["x"] - x1))


def test_is_in_side(point, line, in_side_point):
    anchor = {"x": float(in_side_point[0]), "y": float(in_side_point[1])}
    anchor_sign = get_line_cross_value(anchor, line)
    point_sign = get_line_cross_value(point, line)
    return (anchor_sign * point_sign) >= 0


def point_to_segment_distance(point, line):
    x1, y1 = map(float, line[0])
    x2, y2 = map(float, line[1])
    px = float(point["x"])
    py = float(point["y"])
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = (((px - x1) * dx) + ((py - y1) * dy)) / length_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + (t * dx)
    proj_y = y1 + (t * dy)
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def test_entry_crossing(prev_point, curr_point, line, in_side_point, distance_threshold):
    prev_in = test_is_in_side(prev_point, line, in_side_point)
    curr_in = test_is_in_side(curr_point, line, in_side_point)
    if prev_in or (not curr_in):
        return False
    mid_point = {
        "x": (prev_point["x"] + curr_point["x"]) / 2.0,
        "y": (prev_point["y"] + curr_point["y"]) / 2.0,
    }
    return point_to_segment_distance(mid_point, line) <= distance_threshold


def camera_processing_polygon(camera_cfg):
    return camera_cfg.get("processing_roi") or camera_cfg.get("entry_roi") or camera_cfg.get("track_roi") or []


def find_direction_anchor(records, camera_id, camera_cfg, transition_map, direction_cfg):
    if len(records) < 2:
        return None, "no_track"
    line = camera_cfg.get("entry_line") or []
    in_side_point = camera_cfg.get("in_side_point") or []
    if not line or not in_side_point:
        return None, "missing_manual_entry_line"
    history_window = max(2, as_int((direction_cfg or {}).get("history_window", 6), 6))
    for index in range(1, len(records)):
        start_index = max(0, index - history_window + 1)
        window_rows = records[start_index : index + 1]
        points = [{"x": row["foot_x"], "y": row["foot_y"]} for row in window_rows]
        spatial_history = [
            resolve_spatial_context(camera_id, row.get("foot_x"), row.get("foot_y"), transition_map)
            for row in window_rows
        ]
        result = evaluate_direction(points, line, in_side_point, spatial_history=spatial_history, config=direction_cfg)
        if result["decision"] == "IN":
            return {
                "anchor_index": index,
                "anchor_row": records[index],
                "window_start_index": start_index,
                "window_rows": window_rows,
                "direction_result": result,
            }, "ok"
    return None, "filtered_by_direction"


def clamp_rect(rect, img_width, img_height):
    x = max(0, int(rect["x"]))
    y = max(0, int(rect["y"]))
    width = max(1, min(int(rect["width"]), img_width - x))
    height = max(1, min(int(rect["height"]), img_height - y))
    return {"x": x, "y": y, "width": width, "height": height}


def get_head_rect(record, head_cfg, img_width, img_height):
    width = float(record["width"])
    height = float(record["height"])
    x = float(record["xmin"]) + (float(head_cfg["side_ratio"]) * width)
    y = float(record["ymin"]) + (float(head_cfg["top_ratio"]) * height)
    head_w = width * (1.0 - (2.0 * float(head_cfg["side_ratio"])))
    head_h = height * (float(head_cfg["bottom_ratio"]) - float(head_cfg["top_ratio"]))
    return clamp_rect(
        {
            "x": int(round(x)),
            "y": int(round(y)),
            "width": int(round(head_w)),
            "height": int(round(head_h)),
        },
        img_width,
        img_height,
    )


def write_image_unicode(image_path: Path, image):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {image_path}")
    encoded.tofile(str(image_path))


def load_image_unicode(image_path: Path):
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class FrameSourceCache:
    def __init__(self):
        self._captures = {}

    def close(self):
        for capture in self._captures.values():
            try:
                capture.release()
            except Exception:
                pass
        self._captures.clear()

    def _video_capture(self, video_path: Path):
        key = str(video_path)
        capture = self._captures.get(key)
        if capture is not None:
            return capture
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_path}")
        self._captures[key] = capture
        return capture

    def frame_for_record(self, record, dataset_root: Path):
        source_mode = record.get("frame_source_mode", "image_subsets")
        if source_mode == "video_files" and record.get("video_path"):
            video_path = Path(record["video_path"])
            capture = self._video_capture(video_path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, as_int(record.get("frame_id")))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to decode frame {record.get('frame_id')} from {video_path}")
            return frame
        if record.get("source_image_path"):
            image_path = Path(record["source_image_path"])
        else:
            image_path = dataset_root / record["image_rel_path"]
        frame = load_image_unicode(image_path)
        if frame is None:
            raise RuntimeError(f"Failed to read image source: {image_path}")
        return frame

    def crop_record(self, record, rect, out_path: Path, dataset_root: Path):
        frame = self.frame_for_record(record, dataset_root)
        x = rect["x"]
        y = rect["y"]
        w = rect["width"]
        h = rect["height"]
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            raise RuntimeError(f"Empty crop for {out_path}")
        write_image_unicode(out_path, crop)


def iter_annotation_payloads(annotation_dir: Path, low_load_cfg):
    start_frame = as_int(low_load_cfg.get("start_frame", 0))
    end_frame = low_load_cfg.get("end_frame")
    end_frame = None if end_frame in ("", None) else as_int(end_frame)
    frame_stride = max(1, as_int(low_load_cfg.get("frame_stride", 1), 1))
    enabled = bool(low_load_cfg.get("enabled", False))
    files = sorted(annotation_dir.glob("*.json"))
    for file_path in files:
        frame_id = as_int(file_path.stem)
        if enabled:
            if frame_id < start_frame:
                continue
            if end_frame is not None and frame_id > end_frame:
                continue
            if ((frame_id - start_frame) % frame_stride) != 0:
                continue
        payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
        yield frame_id, payload


def extract_camera_track_rows(dataset_root: Path, camera_id, camera_cfg, fps, video_source: Path, low_load_cfg, frame_source_mode):
    annotation_dir = dataset_root / "annotations_positions"
    polygon = camera_processing_polygon(camera_cfg)
    view_index = as_int(camera_cfg["view_index"])
    rows = []
    for frame_id, objects in iter_annotation_payloads(annotation_dir, low_load_cfg):
        frame_key = f"{frame_id:08d}"
        relative_sec = round(frame_id / float(fps), 3)
        for obj in objects:
            view = obj["views"][view_index]
            if view["xmin"] < 0 or view["ymin"] < 0 or view["xmax"] <= view["xmin"] or view["ymax"] <= view["ymin"]:
                continue
            width = int(view["xmax"] - view["xmin"])
            height = int(view["ymax"] - view["ymin"])
            center_x = round((view["xmin"] + view["xmax"]) / 2.0, 2)
            center_y = round((view["ymin"] + view["ymax"]) / 2.0, 2)
            foot_x = center_x
            foot_y = float(view["ymax"])
            if polygon and not test_point_in_polygon(foot_x, foot_y, polygon):
                continue
            rows.append(
                {
                    "camera_id": camera_id,
                    "role": camera_cfg.get("role", ""),
                    "local_track_id": str(obj["personID"]),
                    "global_gt_id": int(obj["personID"]),
                    "position_id": int(obj["positionID"]),
                    "frame_id": frame_id,
                    "frame_key": frame_key,
                    "relative_sec": relative_sec,
                    "xmin": int(view["xmin"]),
                    "ymin": int(view["ymin"]),
                    "xmax": int(view["xmax"]),
                    "ymax": int(view["ymax"]),
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "center_x": center_x,
                    "center_y": center_y,
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                    "image_rel_path": f"Image_subsets\\{camera_id}\\{frame_key}.png",
                    "video_path": str(video_source),
                    "frame_source_mode": frame_source_mode,
                }
            )
    rows.sort(key=lambda row: (row["frame_id"], row["global_gt_id"]))
    return rows


def group_rows_by_track(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["local_track_id"])].append(row)
    for records in grouped.values():
        records.sort(key=lambda item: item["frame_id"])
    return grouped


DEFAULT_BEST_SHOT_SELECTION = {
    "enabled": False,
    "preferred_subzone_types": ["exit", "interior", "overlap", "transit", "entry"],
    "minimum_frames_after_anchor": {
        "entry": 0,
        "overlap": 0,
        "sequential": 0,
        "weak_link": 0,
    },
}


def _best_shot_cfg(best_shot_cfg):
    merged = dict(DEFAULT_BEST_SHOT_SELECTION)
    merged["minimum_frames_after_anchor"] = dict(DEFAULT_BEST_SHOT_SELECTION["minimum_frames_after_anchor"])
    if not best_shot_cfg:
        return merged
    for key, value in best_shot_cfg.items():
        if key == "minimum_frames_after_anchor" and isinstance(value, dict):
            merged["minimum_frames_after_anchor"].update(value)
        else:
            merged[key] = value
    return merged


def _subzone_priority(subzone_type, cfg):
    preferred = cfg.get("preferred_subzone_types", []) or []
    if subzone_type in preferred:
        return len(preferred) - preferred.index(subzone_type)
    return 0


def select_best_record(records, frame_min=None, frame_max=None, camera_id="", transition_map=None, best_shot_cfg=None, anchor_frame_id=None, relation_type="entry"):
    candidates = []
    for record in records:
        if frame_min is not None and record["frame_id"] < frame_min:
            continue
        if frame_max is not None and record["frame_id"] > frame_max:
            continue
        candidates.append(record)
    if not candidates:
        return None, {}

    cfg = _best_shot_cfg(best_shot_cfg)
    if not cfg.get("enabled") or not camera_id or not transition_map:
        selected = max(candidates, key=lambda row: (row["area"], -row["frame_id"]))
        return selected, {
            "best_shot_strategy": "max_area_in_window",
            "best_shot_reason": "selected_largest_bbox_area_in_window",
            "best_shot_zone_id": "",
            "best_shot_subzone_id": "",
            "best_shot_subzone_type": "",
            "best_shot_frames_after_anchor": as_int(selected["frame_id"]) - as_int(anchor_frame_id, as_int(selected["frame_id"])),
        }

    min_frames_after_anchor = as_int(cfg.get("minimum_frames_after_anchor", {}).get(relation_type, 0))
    ranked = []
    for record in candidates:
        spatial = resolve_spatial_context(camera_id, record.get("foot_x"), record.get("foot_y"), transition_map)
        frames_after_anchor = as_int(record["frame_id"]) - as_int(anchor_frame_id, as_int(record["frame_id"]))
        ranked.append(
            {
                "record": record,
                "spatial": spatial,
                "frames_after_anchor": frames_after_anchor,
                "after_anchor_ok": frames_after_anchor >= min_frames_after_anchor,
                "subzone_priority": _subzone_priority(spatial.get("subzone_type", ""), cfg),
            }
        )
    ranked.sort(
        key=lambda item: (
            1 if item["after_anchor_ok"] else 0,
            item["subzone_priority"],
            item["record"]["area"],
            item["frames_after_anchor"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    record = selected["record"]
    spatial = selected["spatial"]
    reason_bits = [
        f"after_anchor_ok={str(selected['after_anchor_ok']).lower()}",
        f"frames_after_anchor={selected['frames_after_anchor']}",
        f"subzone_type={spatial.get('subzone_type', '') or 'none'}",
        f"bbox_area={record['area']}",
    ]
    return record, {
        "best_shot_strategy": "line_aware_subzone_priority",
        "best_shot_reason": ";".join(reason_bits),
        "best_shot_zone_id": spatial.get("zone_id", ""),
        "best_shot_subzone_id": spatial.get("subzone_id", ""),
        "best_shot_subzone_type": spatial.get("subzone_type", ""),
        "best_shot_frames_after_anchor": selected["frames_after_anchor"],
    }


def crop_event_from_record(frame_cache: FrameSourceCache, dataset_root: Path, crop_root: Path, event_id, record, head_cfg):
    frame = frame_cache.frame_for_record(record, dataset_root)
    img_h, img_w = frame.shape[:2]
    body_rect = clamp_rect(
        {"x": record["xmin"], "y": record["ymin"], "width": record["width"], "height": record["height"]},
        img_w,
        img_h,
    )
    head_rect = get_head_rect(record, head_cfg, img_w, img_h)
    camera_dir = crop_root / record["camera_id"]
    body_path = camera_dir / f"{event_id}_body.png"
    head_path = camera_dir / f"{event_id}_head.png"
    frame_cache.crop_record(record, body_rect, body_path, dataset_root)
    frame_cache.crop_record(record, head_rect, head_path, dataset_root)
    return body_path, head_path


def build_entry_events(
    dataset_root: Path,
    wildtrack_config,
    track_rows_by_camera,
    frame_cache: FrameSourceCache,
    crop_root: Path,
    transition_map,
):
    best_shot_window = int(wildtrack_config["best_shot_window_frames"])
    head_cfg = wildtrack_config["head_crop"]
    best_shot_cfg = wildtrack_config.get("best_shot_selection", {})
    direction_cfg = wildtrack_config.get("direction_filter", {})
    event_rows = []
    queue_rows = []
    event_assignment_rows = []

    for camera_id in wildtrack_config["selected_cameras"]:
        camera_cfg = wildtrack_config["cameras"][camera_id]
        if camera_cfg.get("role") != "entry":
            continue
        track_map = group_rows_by_track(track_rows_by_camera.get(camera_id, []))
        for local_track_id, records in track_map.items():
            if len(records) < 2:
                continue
            anchor, anchor_reason = find_direction_anchor(records, camera_id, camera_cfg, transition_map, direction_cfg)
            if anchor is None:
                continue
            crossing_row = anchor["anchor_row"]
            direction_result = anchor["direction_result"]
            best_record, best_shot_meta = select_best_record(
                records,
                max(crossing_row["frame_id"] - 20, 0),
                crossing_row["frame_id"] + best_shot_window,
                camera_id=camera_id,
                transition_map=transition_map,
                best_shot_cfg=best_shot_cfg,
                anchor_frame_id=crossing_row["frame_id"],
                relation_type="entry",
            )
            if best_record is None:
                continue
            event_id = f"IN_{camera_id}_{local_track_id}_{crossing_row['frame_id']:08d}"
            body_path, head_path = crop_event_from_record(frame_cache, dataset_root, crop_root, event_id, best_record, head_cfg)
            spatial = resolve_spatial_context(camera_id, best_record["foot_x"], best_record["foot_y"], transition_map)
            event = {
                "event_id": event_id,
                "event_type": "ENTRY_IN",
                "camera_id": camera_id,
                "camera_role": camera_cfg.get("role", ""),
                "local_track_id": str(local_track_id),
                "global_gt_id": int(best_record["global_gt_id"]),
                "frame_id": int(crossing_row["frame_id"]),
                "relative_sec": float(crossing_row["relative_sec"]),
                "best_shot_frame": int(best_record["frame_id"]),
                "best_shot_sec": float(best_record["relative_sec"]),
                "best_head_crop": str(head_path),
                "best_body_crop": str(body_path),
                "direction": "IN",
                "direction_reason": direction_result.get("reason", ""),
                "direction_history_points": direction_result.get("history_points", 0),
                "direction_momentum_px": direction_result.get("momentum_px", 0.0),
                "direction_inside_ratio": direction_result.get("inside_ratio", 0.0),
                "direction_cross_in": direction_result.get("cross_in", False),
                "direction_zone_transition_ok": direction_result.get("zone_transition_ok", False),
                "source_video": best_record.get("video_path", ""),
                "source_frame_idx": int(best_record["frame_id"]),
                "bbox_xmin": int(best_record["xmin"]),
                "bbox_ymin": int(best_record["ymin"]),
                "bbox_xmax": int(best_record["xmax"]),
                "bbox_ymax": int(best_record["ymax"]),
                "bbox_width": int(best_record["width"]),
                "bbox_height": int(best_record["height"]),
                "bbox_area": int(best_record["area"]),
                "foot_x": best_record["foot_x"],
                "foot_y": best_record["foot_y"],
                "zone_id": spatial["zone_id"],
                "zone_type": spatial["zone_type"],
                "zone_reason": spatial["zone_reason"],
                "zone_fallback_used": spatial["zone_fallback_used"],
                "subzone_id": spatial["subzone_id"],
                "subzone_type": spatial["subzone_type"],
                "subzone_reason": spatial["subzone_reason"],
                "subzone_fallback_used": spatial["subzone_fallback_used"],
                "matched_zone_region_id": spatial["matched_zone_region_id"],
                "matched_subzone_region_id": spatial["matched_subzone_region_id"],
                "assignment_point_x": spatial["assignment_point_x"],
                "assignment_point_y": spatial["assignment_point_y"],
                "best_shot_strategy": best_shot_meta.get("best_shot_strategy", ""),
                "best_shot_reason": best_shot_meta.get("best_shot_reason", ""),
                "best_shot_zone_id": best_shot_meta.get("best_shot_zone_id", ""),
                "best_shot_subzone_id": best_shot_meta.get("best_shot_subzone_id", ""),
                "best_shot_subzone_type": best_shot_meta.get("best_shot_subzone_type", ""),
                "best_shot_frames_after_anchor": best_shot_meta.get("best_shot_frames_after_anchor", ""),
            }
            event_rows.append(event)
            queue_rows.append(
                {
                    "event_id": event_id,
                    "queue_stage": "face_matching",
                    "event_type": "ENTRY_IN",
                    "camera_id": camera_id,
                    "camera_role": camera_cfg.get("role", ""),
                    "global_gt_id": int(best_record["global_gt_id"]),
                    "local_track_id": str(local_track_id),
                    "frame_id": int(crossing_row["frame_id"]),
                    "relative_sec": float(crossing_row["relative_sec"]),
                    "best_head_crop": str(head_path),
                    "best_body_crop": str(body_path),
                    "direction": "IN",
                    "direction_reason": direction_result.get("reason", ""),
                    "direction_history_points": direction_result.get("history_points", 0),
                    "direction_momentum_px": direction_result.get("momentum_px", 0.0),
                    "direction_inside_ratio": direction_result.get("inside_ratio", 0.0),
                    "direction_cross_in": direction_result.get("cross_in", False),
                    "direction_zone_transition_ok": direction_result.get("zone_transition_ok", False),
                    "zone_id": spatial["zone_id"],
                    "zone_type": spatial["zone_type"],
                    "subzone_id": spatial["subzone_id"],
                    "subzone_type": spatial["subzone_type"],
                    "foot_x": best_record["foot_x"],
                    "foot_y": best_record["foot_y"],
                    "best_shot_strategy": best_shot_meta.get("best_shot_strategy", ""),
                    "best_shot_reason": best_shot_meta.get("best_shot_reason", ""),
                    "best_shot_subzone_id": best_shot_meta.get("best_shot_subzone_id", ""),
                    "best_shot_subzone_type": best_shot_meta.get("best_shot_subzone_type", ""),
                    "best_shot_frames_after_anchor": best_shot_meta.get("best_shot_frames_after_anchor", ""),
                    "matched_known_id": "",
                    "matched_known_score": "",
                    "unknown_global_id": "",
                    "identity_status": "pending",
                    "next_step": "match_face_then_assign_known_or_unknown",
                }
            )
            event_assignment_rows.append(
                {
                    "run_mode": "offline_entry_event_builder",
                    "event_id": event_id,
                    "event_type": "ENTRY_IN",
                    "camera_id": camera_id,
                    "relative_sec": float(crossing_row["relative_sec"]),
                    "zone_id": spatial["zone_id"],
                    "zone_type": spatial["zone_type"],
                    "subzone_id": spatial["subzone_id"],
                    "subzone_type": spatial["subzone_type"],
                    "assignment_point_x": spatial["assignment_point_x"],
                    "assignment_point_y": spatial["assignment_point_y"],
                    "matched_zone_region_id": spatial["matched_zone_region_id"],
                    "matched_subzone_region_id": spatial["matched_subzone_region_id"],
                    "zone_reason": spatial["zone_reason"],
                    "subzone_reason": spatial["subzone_reason"],
                    "zone_fallback_used": spatial["zone_fallback_used"],
                    "subzone_fallback_used": spatial["subzone_fallback_used"],
                    "direction_reason": direction_result.get("reason", ""),
                    "direction_history_points": direction_result.get("history_points", 0),
                    "direction_momentum_px": direction_result.get("momentum_px", 0.0),
                    "direction_inside_ratio": direction_result.get("inside_ratio", 0.0),
                    "best_shot_strategy": best_shot_meta.get("best_shot_strategy", ""),
                    "best_shot_reason": best_shot_meta.get("best_shot_reason", ""),
                    "best_shot_subzone_id": best_shot_meta.get("best_shot_subzone_id", ""),
                    "best_shot_subzone_type": best_shot_meta.get("best_shot_subzone_type", ""),
                    "best_shot_frames_after_anchor": best_shot_meta.get("best_shot_frames_after_anchor", ""),
                }
            )
    event_rows.sort(key=lambda row: (row["relative_sec"], row["camera_id"], row["local_track_id"]))
    queue_rows.sort(key=lambda row: (row["relative_sec"], row["camera_id"], row["local_track_id"]))
    return event_rows, queue_rows, event_assignment_rows


def build_global_gt_summary(track_rows):
    grouped = {}
    for row in track_rows:
        key = str(row["global_gt_id"])
        if key not in grouped:
            grouped[key] = {
                "global_gt_id": int(row["global_gt_id"]),
                "cameras_seen": set(),
                "first_frame": int(row["frame_id"]),
                "last_frame": int(row["frame_id"]),
                "first_sec": float(row["relative_sec"]),
                "last_sec": float(row["relative_sec"]),
            }
        item = grouped[key]
        item["cameras_seen"].add(row["camera_id"])
        if row["frame_id"] < item["first_frame"]:
            item["first_frame"] = int(row["frame_id"])
            item["first_sec"] = float(row["relative_sec"])
        if row["frame_id"] > item["last_frame"]:
            item["last_frame"] = int(row["frame_id"])
            item["last_sec"] = float(row["relative_sec"])
    rows = []
    for gt_id in sorted(grouped.keys(), key=lambda value: int(value)):
        item = grouped[gt_id]
        rows.append(
            {
                "global_gt_id": item["global_gt_id"],
                "cameras_seen": ",".join(sorted(item["cameras_seen"])),
                "first_frame": item["first_frame"],
                "last_frame": item["last_frame"],
                "first_sec": item["first_sec"],
                "last_sec": item["last_sec"],
            }
        )
    return rows


def materialize_stage_inputs(project_root: Path, dataset_root: Path, wildtrack_config_path: Path, wildtrack_config, output_root: Path, track_rows_by_camera, frame_source_mode, low_load_cfg, transition_map):
    tracks_dir = output_root / "tracks"
    events_dir = output_root / "events"
    crops_dir = output_root / "crops"
    summaries_dir = output_root / "summaries"
    audit_dir = output_root / "audit"
    for directory in (tracks_dir, events_dir, crops_dir, summaries_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_track_rows = []
    for camera_id in wildtrack_config["selected_cameras"]:
        rows = track_rows_by_camera.get(camera_id, [])
        all_track_rows.extend(rows)
        write_csv(tracks_dir / f"{camera_id}_tracks.csv", rows, fieldnames_for_rows(rows))

    all_track_rows.sort(key=lambda row: (row["frame_id"], row["camera_id"], row["global_gt_id"]))
    write_csv(tracks_dir / "all_tracks_filtered.csv", all_track_rows, fieldnames_for_rows(all_track_rows))

    frame_cache = FrameSourceCache()
    try:
        entry_events, identity_queue_rows, assignment_rows = build_entry_events(
            dataset_root,
            wildtrack_config,
            track_rows_by_camera,
            frame_cache,
            crops_dir,
            transition_map,
        )
    finally:
        frame_cache.close()

    write_csv(events_dir / "entry_in_events.csv", entry_events, fieldnames_for_rows(entry_events, ["event_id"]))
    save_json(events_dir / "entry_in_events.json", entry_events)
    write_csv(
        events_dir / "identity_resolution_queue.csv",
        identity_queue_rows,
        fieldnames_for_rows(identity_queue_rows, ["event_id"]),
    )
    save_json(events_dir / "identity_resolution_queue.json", identity_queue_rows)

    global_gt_summary_rows = build_global_gt_summary(all_track_rows)
    write_csv(summaries_dir / "global_gt_summary.csv", global_gt_summary_rows, fieldnames_for_rows(global_gt_summary_rows))
    write_csv(
        audit_dir / "entry_event_assignment_audit.csv",
        assignment_rows,
        fieldnames_for_rows(assignment_rows, ["run_mode", "event_id"]),
    )
    stage_summary = {
        "pipeline_backend": "wildtrack_gt_annotations",
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "wildtrack_config_path": str(wildtrack_config_path),
        "frame_source_mode": frame_source_mode,
        "low_load": low_load_cfg,
        "selected_cameras": list(wildtrack_config["selected_cameras"]),
        "filtered_track_rows": len(all_track_rows),
        "entry_in_events": len(entry_events),
        "identity_queue_rows": len(identity_queue_rows),
        "per_camera_rows": {camera_id: len(track_rows_by_camera[camera_id]) for camera_id in wildtrack_config["selected_cameras"]},
        "notes": [
            "This offline stage uses Wildtrack annotations as a GT-backed detect/track provider for the thesis demo baseline.",
            "Direction filtering is applied before face matching and only ENTRY_IN candidates are exported to the identity queue.",
            "Frame crops are taken from the configured frame source mode for reproducible best-shot generation.",
        ],
    }
    save_json(summaries_dir / "stage_input_summary.json", stage_summary)
    save_json(summaries_dir / "wildtrack_demo_config.runtime.json", wildtrack_config)

    return {
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "wildtrack_config_path": str(wildtrack_config_path),
        "output_root": str(output_root),
        "tracks_csv": str(tracks_dir / "all_tracks_filtered.csv"),
        "identity_queue_csv": str(events_dir / "identity_resolution_queue.csv"),
        "entry_events_csv": str(events_dir / "entry_in_events.csv"),
        "stage_summary_json": str(summaries_dir / "stage_input_summary.json"),
        "stage_assignment_audit_csv": str(audit_dir / "entry_event_assignment_audit.csv"),
    }


def build_entry_anchor_packets(camera_id, camera_cfg, camera_rows, line_threshold, transition_map):
    packets = []
    track_map = group_rows_by_track(camera_rows)
    direction_cfg = camera_cfg.get("direction_filter", {})
    for local_track_id, records in track_map.items():
        if len(records) < 2:
            continue
        anchor, _anchor_reason = find_direction_anchor(records, camera_id, camera_cfg, transition_map, direction_cfg)
        if anchor is None:
            continue
        curr_row = anchor["anchor_row"]
        direction_result = anchor["direction_result"]
        spatial = resolve_spatial_context(camera_id, curr_row["foot_x"], curr_row["foot_y"], transition_map)
        packets.append(
            {
                "packet_type": "entry_anchor",
                "camera_id": camera_id,
                "frame_idx": int(curr_row["frame_id"]),
                "relative_sec": float(curr_row["relative_sec"]),
                "local_track_id": str(local_track_id),
                "global_gt_id": int(curr_row["global_gt_id"]),
                "direction": "IN",
                "direction_reason": direction_result.get("reason", ""),
                "direction_history_points": direction_result.get("history_points", 0),
                "direction_momentum_px": direction_result.get("momentum_px", 0.0),
                "direction_inside_ratio": direction_result.get("inside_ratio", 0.0),
                "bbox_xmin": int(curr_row["xmin"]),
                "bbox_ymin": int(curr_row["ymin"]),
                "bbox_xmax": int(curr_row["xmax"]),
                "bbox_ymax": int(curr_row["ymax"]),
                "bbox_area": int(curr_row["area"]),
                "foot_x": float(curr_row["foot_x"]),
                "foot_y": float(curr_row["foot_y"]),
                "zone_id": spatial["zone_id"],
                "subzone_id": spatial["subzone_id"],
                "zone_type": spatial["zone_type"],
                "subzone_type": spatial["subzone_type"],
                "crop_reference": {
                    "frame_source_mode": curr_row.get("frame_source_mode", "image_subsets"),
                    "video_path": curr_row.get("video_path", ""),
                    "image_rel_path": curr_row.get("image_rel_path", ""),
                },
            }
        )
    packets.sort(key=lambda row: (row["relative_sec"], row["camera_id"], row["local_track_id"]))
    return packets


def build_offline_stage_inputs(offline_config, transition_map, wildtrack_config_override=None):
    project_root = Path(offline_config["project_root"]).resolve()
    dataset_root = Path(offline_config["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = (project_root / dataset_root).resolve()
    wildtrack_config_path = Path(offline_config["wildtrack_demo_config"])
    if not wildtrack_config_path.is_absolute():
        wildtrack_config_path = (project_root / wildtrack_config_path).resolve()
    wildtrack_config = wildtrack_config_override or load_json(wildtrack_config_path)
    output_root = Path(offline_config["output_root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()

    frame_source_mode = offline_config.get("frame_source_mode", "video_files")
    video_sources = {}
    for camera_id, video_path in (offline_config["dataset"].get("video_sources", {}) or {}).items():
        resolved = Path(video_path)
        if not resolved.is_absolute():
            resolved = (project_root / resolved).resolve()
        video_sources[camera_id] = resolved

    fps = float(wildtrack_config["assumed_video_fps"])
    low_load_cfg = offline_config.get("low_load", {})
    track_rows_by_camera = {}
    for camera_id in wildtrack_config["selected_cameras"]:
        if camera_id not in video_sources:
            raise RuntimeError(f"Missing video source for {camera_id} in offline pipeline config.")
        track_rows_by_camera[camera_id] = extract_camera_track_rows(
            dataset_root,
            camera_id,
            wildtrack_config["cameras"][camera_id],
            fps,
            video_sources[camera_id],
            low_load_cfg,
            frame_source_mode,
        )

    result = materialize_stage_inputs(
        project_root,
        dataset_root,
        wildtrack_config_path,
        wildtrack_config,
        output_root,
        track_rows_by_camera,
        frame_source_mode,
        low_load_cfg,
        transition_map,
    )
    result["video_sources"] = {camera_id: str(path) for camera_id, path in video_sources.items()}
    return result
