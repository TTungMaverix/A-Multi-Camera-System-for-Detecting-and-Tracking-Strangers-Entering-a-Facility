import copy
import csv
import hashlib
import json
import math
import time
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
            capture.set(cv2.CAP_PROP_POS_FRAMES, as_int(record.get("source_frame_id_actual", record.get("frame_id"))))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Failed to decode frame {record.get('source_frame_id_actual', record.get('frame_id'))} from {video_path}"
                )
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


def _track_bbox(row):
    return {
        "xmin": float(row["xmin"]),
        "ymin": float(row["ymin"]),
        "xmax": float(row["xmax"]),
        "ymax": float(row["ymax"]),
    }


def _bbox_iou(box_a, box_b):
    inter_x1 = max(float(box_a["xmin"]), float(box_b["xmin"]))
    inter_y1 = max(float(box_a["ymin"]), float(box_b["ymin"]))
    inter_x2 = min(float(box_a["xmax"]), float(box_b["xmax"]))
    inter_y2 = min(float(box_a["ymax"]), float(box_b["ymax"]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(1.0, (float(box_a["xmax"]) - float(box_a["xmin"])) * (float(box_a["ymax"]) - float(box_a["ymin"])))
    area_b = max(1.0, (float(box_b["xmax"]) - float(box_b["xmin"])) * (float(box_b["ymax"]) - float(box_b["ymin"])))
    return inter_area / max(1.0, area_a + area_b - inter_area)


def _bbox_center(box):
    return (
        (float(box["xmin"]) + float(box["xmax"])) / 2.0,
        (float(box["ymin"]) + float(box["ymax"])) / 2.0,
    )


def _bbox_diagonal(box):
    width = max(1.0, float(box["xmax"]) - float(box["xmin"]))
    height = max(1.0, float(box["ymax"]) - float(box["ymin"]))
    return math.sqrt((width * width) + (height * height))


def _shift_bbox(box, delta_x, delta_y):
    return {
        "xmin": float(box["xmin"]) + float(delta_x),
        "ymin": float(box["ymin"]) + float(delta_y),
        "xmax": float(box["xmax"]) + float(delta_x),
        "ymax": float(box["ymax"]) + float(delta_y),
    }


def _tracklet_summary(track_id, rows):
    ordered = sorted(rows, key=lambda item: (item["frame_id"], item.get("source_frame_id_actual", item["frame_id"])))
    centers = [_bbox_center(_track_bbox(row)) for row in ordered[-2:]]
    return {
        "track_id": str(track_id),
        "rows": ordered,
        "start_frame": int(ordered[0]["frame_id"]),
        "end_frame": int(ordered[-1]["frame_id"]),
        "start_bbox": _track_bbox(ordered[0]),
        "end_bbox": _track_bbox(ordered[-1]),
        "start_center": _bbox_center(_track_bbox(ordered[0])),
        "end_center": _bbox_center(_track_bbox(ordered[-1])),
        "last_centers": centers,
    }


def _predict_tracklet_bbox(summary, gap_frames):
    # ByteTrack fragmentation after short occlusion often creates a new ID a few frames later.
    # We extrapolate the last observed bbox with a constant-velocity model from the final two
    # centers. This is deliberately simple: it keeps the linker cheap while still checking
    # motion continuity instead of relying on IoU alone.
    velocity_x = 0.0
    velocity_y = 0.0
    centers = summary.get("last_centers", [])
    if len(centers) >= 2:
        velocity_x = centers[-1][0] - centers[-2][0]
        velocity_y = centers[-1][1] - centers[-2][1]
    delta_x = velocity_x * float(gap_frames)
    delta_y = velocity_y * float(gap_frames)
    predicted_bbox = _shift_bbox(summary["end_bbox"], delta_x, delta_y)
    predicted_center = _bbox_center(predicted_bbox)
    return predicted_bbox, predicted_center, (velocity_x, velocity_y)


def link_short_gap_tracklets(track_rows, linking_cfg=None):
    default_cfg = {
        "enabled": True,
        "max_gap_frames": 12,
        "min_iou": 0.15,
        "max_normalized_center_distance": 0.65,
        "min_motion_continuity": 0.25,
        "score_weights": {
            "iou": 0.65,
            "motion": 0.30,
            "gap_penalty": 0.05,
        },
    }
    cfg = copy.deepcopy(default_cfg)
    for key, value in (linking_cfg or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value

    if not cfg.get("enabled", True) or len(track_rows) < 2:
        return track_rows, {
            "enabled": bool(cfg.get("enabled", True)),
            "original_tracklet_count": len({str(row.get("local_track_id", "")) for row in track_rows}),
            "linked_tracklet_count": len({str(row.get("local_track_id", "")) for row in track_rows}),
            "links_applied": 0,
            "candidate_pairs_considered": 0,
            "link_details": [],
        }

    grouped = defaultdict(list)
    for row in track_rows:
        grouped[str(row.get("local_track_id", ""))].append(row)
    summaries = [_tracklet_summary(track_id, rows) for track_id, rows in grouped.items()]
    summaries.sort(key=lambda item: (item["start_frame"], item["end_frame"], item["track_id"]))

    canonical_ids = {summary["track_id"]: summary["track_id"] for summary in summaries}
    predecessor_taken = set()
    candidate_pairs_considered = 0
    links_applied = 0
    link_details = []

    for current in summaries:
        best_match = None
        for previous in summaries:
            if previous["track_id"] == current["track_id"]:
                continue
            if previous["track_id"] in predecessor_taken:
                continue
            gap_frames = current["start_frame"] - previous["end_frame"]
            if gap_frames <= 0 or gap_frames > int(cfg["max_gap_frames"]):
                continue

            candidate_pairs_considered += 1
            predicted_bbox, predicted_center, velocity = _predict_tracklet_bbox(previous, gap_frames)
            raw_iou = _bbox_iou(previous["end_bbox"], current["start_bbox"])
            predicted_iou = _bbox_iou(predicted_bbox, current["start_bbox"])
            effective_iou = max(raw_iou, predicted_iou)

            start_center = current["start_center"]
            center_distance = math.sqrt(
                ((predicted_center[0] - start_center[0]) ** 2) + ((predicted_center[1] - start_center[1]) ** 2)
            )
            normalized_center_distance = center_distance / max(
                1.0,
                _bbox_diagonal(previous["end_bbox"]),
                _bbox_diagonal(current["start_bbox"]),
            )
            motion_continuity = max(
                0.0,
                1.0 - (normalized_center_distance / max(float(cfg["max_normalized_center_distance"]), 1e-6)),
            )

            if effective_iou < float(cfg["min_iou"]) and normalized_center_distance > float(cfg["max_normalized_center_distance"]):
                continue
            if motion_continuity < float(cfg["min_motion_continuity"]):
                continue

            score = (
                (float(cfg["score_weights"]["iou"]) * effective_iou)
                + (float(cfg["score_weights"]["motion"]) * motion_continuity)
                - (float(cfg["score_weights"]["gap_penalty"]) * (gap_frames / max(1.0, float(cfg["max_gap_frames"]))))
            )
            candidate = {
                "previous_track_id": previous["track_id"],
                "current_track_id": current["track_id"],
                "gap_frames": gap_frames,
                "raw_iou": round(raw_iou, 4),
                "predicted_iou": round(predicted_iou, 4),
                "effective_iou": round(effective_iou, 4),
                "center_distance_px": round(center_distance, 4),
                "normalized_center_distance": round(normalized_center_distance, 4),
                "motion_continuity": round(motion_continuity, 4),
                "velocity_px": [round(float(velocity[0]), 4), round(float(velocity[1]), 4)],
                "score": round(score, 4),
            }
            if best_match is None or candidate["score"] > best_match["score"]:
                best_match = candidate

        if best_match is None:
            continue

        root_track_id = canonical_ids.get(best_match["previous_track_id"], best_match["previous_track_id"])
        canonical_ids[current["track_id"]] = root_track_id
        predecessor_taken.add(best_match["previous_track_id"])
        links_applied += 1
        best_match["linked_track_id"] = root_track_id
        link_details.append(best_match)

    rewritten_rows = []
    for row in track_rows:
        original_track_id = str(row.get("local_track_id", ""))
        linked_track_id = canonical_ids.get(original_track_id, original_track_id)
        rewritten = dict(row)
        rewritten["bytetrack_track_id"] = original_track_id
        rewritten["linked_track_id"] = linked_track_id
        rewritten["tracklet_linked"] = linked_track_id != original_track_id
        rewritten["source_track_surrogate_id"] = linked_track_id
        rewritten["local_track_id"] = linked_track_id
        if linked_track_id.isdigit():
            rewritten["global_gt_id"] = int(linked_track_id)
        rewritten_rows.append(rewritten)

    rewritten_rows.sort(key=lambda row: (row["frame_id"], row["global_gt_id"], row.get("source_frame_id_actual", row["frame_id"])))
    runtime_info = {
        "enabled": True,
        "original_tracklet_count": len(summaries),
        "linked_tracklet_count": len(set(canonical_ids.values())),
        "links_applied": links_applied,
        "candidate_pairs_considered": candidate_pairs_considered,
        "max_gap_frames": int(cfg["max_gap_frames"]),
        "min_iou": float(cfg["min_iou"]),
        "max_normalized_center_distance": float(cfg["max_normalized_center_distance"]),
        "min_motion_continuity": float(cfg["min_motion_continuity"]),
        "link_details": link_details[:20],
        "notes": [
            "Short-gap linking runs immediately after ByteTrack and before downstream event generation.",
            "A candidate link must survive both spatial continuity checks and a bounded gap window.",
            "IoU is measured between the last bbox of the old tracklet and the first bbox of the new one, with an additional predicted-IoU check after constant-velocity extrapolation.",
        ],
    }
    return rewritten_rows, runtime_info


def _merge_cache_cfg(cache_cfg=None):
    merged = {
        "enabled": False,
        "use_cache": True,
        "refresh_cache": False,
        "cache_dir": "",
    }
    for key, value in (cache_cfg or {}).items():
        merged[key] = value
    return merged


def _resolve_cache_dir(cache_cfg, default_root: Path):
    cache_dir = cache_cfg.get("cache_dir", "") or ""
    if cache_dir:
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = (default_root / cache_path).resolve()
        return cache_path
    return (default_root / ".cache" / "detector_tracker").resolve()


def _cache_signature(
    camera_id,
    video_source: Path,
    low_load_cfg,
    inference_cfg,
    linking_cfg,
    actual_video_fps,
    target_timeline_fps,
):
    return {
        "camera_id": camera_id,
        "video_source": str(video_source),
        "start_frame": as_int(low_load_cfg.get("start_frame", 0)),
        "end_frame": (
            ""
            if low_load_cfg.get("end_frame") in ("", None)
            else as_int(low_load_cfg.get("end_frame"))
        ),
        "frame_stride": max(1, as_int(low_load_cfg.get("frame_stride", 1), 1)),
        "detector_model": inference_cfg.get("detector_model", "yolov8n.pt"),
        "tracker": inference_cfg.get("tracker", "bytetrack.yaml"),
        "device": str(inference_cfg.get("device", "cpu")),
        "conf_threshold": float(inference_cfg.get("conf_threshold", 0.25)),
        "iou_threshold": float(inference_cfg.get("iou_threshold", 0.5)),
        "imgsz": int(inference_cfg.get("imgsz", 640)),
        "person_class_id": int(inference_cfg.get("person_class_id", 0)),
        "target_timeline_fps": round(float(target_timeline_fps), 4),
        "actual_video_fps": round(float(actual_video_fps or 0.0), 4),
        "tracklet_linking": copy.deepcopy(linking_cfg or {}),
    }


def _cache_key(signature):
    payload = json.dumps(signature, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_paths(cache_dir: Path, signature):
    cache_key = _cache_key(signature)
    cache_root = cache_dir / cache_key
    return {
        "cache_key": cache_key,
        "cache_root": cache_root,
        "metadata_json": cache_root / "metadata.json",
        "track_rows_json": cache_root / "track_rows.json",
        "runtime_json": cache_root / "runtime_info.json",
    }


def load_inference_track_cache(cache_dir: Path, signature):
    paths = _cache_paths(cache_dir, signature)
    started = time.perf_counter()
    metadata_path = paths["metadata_json"]
    track_rows_path = paths["track_rows_json"]
    runtime_path = paths["runtime_json"]
    if not metadata_path.exists() or not track_rows_path.exists():
        return None, {
            "cache_enabled": True,
            "cache_hit": False,
            "cache_key": paths["cache_key"],
            "cache_root": str(paths["cache_root"]),
            "cache_miss_reason": "cache_files_missing",
            "cache_load_time_sec": round(max(time.perf_counter() - started, 1e-9), 6),
        }
    metadata = load_json(metadata_path)
    if metadata.get("signature") != signature:
        return None, {
            "cache_enabled": True,
            "cache_hit": False,
            "cache_key": paths["cache_key"],
            "cache_root": str(paths["cache_root"]),
            "cache_miss_reason": "signature_mismatch",
            "cache_load_time_sec": round(max(time.perf_counter() - started, 1e-9), 6),
        }
    runtime_info = load_json(runtime_path) if runtime_path.exists() else {}
    track_rows = load_json(track_rows_path)
    return track_rows, {
        "cache_enabled": True,
        "cache_hit": True,
        "cache_key": paths["cache_key"],
        "cache_root": str(paths["cache_root"]),
        "metadata_json": str(metadata_path),
        "track_rows_json": str(track_rows_path),
        "runtime_json": str(runtime_path),
        "cache_created_at": metadata.get("created_at_utc", ""),
        "cache_load_time_sec": round(max(time.perf_counter() - started, 1e-9), 6),
        "cached_runtime_elapsed_sec": runtime_info.get("elapsed_sec", 0.0),
    }


def write_inference_track_cache(cache_dir: Path, signature, track_rows, runtime_info):
    paths = _cache_paths(cache_dir, signature)
    paths["cache_root"].mkdir(parents=True, exist_ok=True)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(
        paths["metadata_json"],
        {
            "signature": signature,
            "created_at_utc": created_at,
        },
    )
    save_json(paths["track_rows_json"], track_rows)
    save_json(paths["runtime_json"], runtime_info)
    return {
        "cache_key": paths["cache_key"],
        "cache_root": str(paths["cache_root"]),
        "metadata_json": str(paths["metadata_json"]),
        "track_rows_json": str(paths["track_rows_json"]),
        "runtime_json": str(paths["runtime_json"]),
        "cache_created_at": created_at,
    }


def extract_inference_track_rows(
    camera_id,
    camera_cfg,
    fps,
    video_source: Path,
    low_load_cfg,
    frame_source_mode,
    inference_cfg,
):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "single_source sequential inference replay requires ultralytics to be installed in the active runtime."
        ) from exc

    model_name = inference_cfg.get("detector_model", "yolov8n.pt")
    tracker_name = inference_cfg.get("tracker", "bytetrack.yaml")
    person_class_id = int(inference_cfg.get("person_class_id", 0))
    conf_threshold = float(inference_cfg.get("conf_threshold", 0.25))
    iou_threshold = float(inference_cfg.get("iou_threshold", 0.5))
    imgsz = int(inference_cfg.get("imgsz", 640))
    device = str(inference_cfg.get("device", "cpu"))
    verbose = bool(inference_cfg.get("verbose", False))
    target_timeline_fps = float(inference_cfg.get("target_timeline_fps", fps))
    polygon = camera_processing_polygon(camera_cfg)
    start_frame = as_int(low_load_cfg.get("start_frame", 0))
    end_frame_raw = low_load_cfg.get("end_frame")
    end_frame = None if end_frame_raw in ("", None) else as_int(end_frame_raw)
    frame_stride = max(1, as_int(low_load_cfg.get("frame_stride", 1), 1))

    capture = cv2.VideoCapture(str(video_source))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open inference replay video source: {video_source}")
    actual_video_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps or 0.0)
    capture.release()

    linking_cfg = (inference_cfg or {}).get("tracklet_linking")
    cache_cfg = _merge_cache_cfg((inference_cfg or {}).get("cache"))
    cache_dir = _resolve_cache_dir(cache_cfg, video_source.parent)
    signature = _cache_signature(
        camera_id,
        video_source,
        low_load_cfg,
        inference_cfg,
        linking_cfg,
        actual_video_fps,
        target_timeline_fps,
    )
    cache_runtime = {
        "cache_enabled": bool(cache_cfg.get("enabled", False)),
        "cache_hit": False,
        "cache_key": _cache_key(signature),
        "cache_root": str(_cache_paths(cache_dir, signature)["cache_root"]),
        "cache_miss_reason": "cache_disabled",
        "cache_load_time_sec": 0.0,
        "time_saved_sec": 0.0,
    }
    if cache_cfg.get("enabled", False) and cache_cfg.get("use_cache", True) and not cache_cfg.get("refresh_cache", False):
        cached_rows, cache_runtime = load_inference_track_cache(cache_dir, signature)
        if cached_rows is not None:
            cached_runtime_info = load_json(Path(cache_runtime["runtime_json"])) if Path(cache_runtime["runtime_json"]).exists() else {}
            runtime_info = dict(cached_runtime_info)
            runtime_info["cache_runtime"] = {
                **cache_runtime,
                "time_saved_sec": round(float(cached_runtime_info.get("elapsed_sec", 0.0)) - float(cache_runtime.get("cache_load_time_sec", 0.0)), 3),
            }
            runtime_info.setdefault("notes", []).append(
                "Detector/tracker rows were loaded from cache generated by true inference, not from GT-backed annotations."
            )
            return cached_rows, runtime_info

    model = YOLO(model_name)
    capture = cv2.VideoCapture(str(video_source))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open inference replay video source: {video_source}")
    track_rows = []
    processed_frames = 0
    raw_detection_count = 0
    emitted_track_rows = 0
    runtime_started = time.perf_counter()
    frame_id = -1
    synthetic_track_id = 1000000
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_id += 1
            if frame_id < start_frame:
                continue
            if end_frame is not None and frame_id > end_frame:
                break
            if ((frame_id - start_frame) % frame_stride) != 0:
                continue

            processed_frames += 1
            result_list = model.track(
                frame,
                persist=True,
                tracker=tracker_name,
                classes=[person_class_id],
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=verbose,
            )
            if not result_list:
                continue
            result = result_list[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy().tolist()
            ids = boxes.id.cpu().numpy().tolist() if boxes.id is not None else [None] * len(xyxy)
            confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
            raw_detection_count += len(xyxy)
            for box, track_id_value, det_score in zip(xyxy, ids, confs):
                xmin, ymin, xmax, ymax = [int(round(value)) for value in box]
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = max(xmin + 1, xmax)
                ymax = max(ymin + 1, ymax)
                width = xmax - xmin
                height = ymax - ymin
                if width <= 0 or height <= 0:
                    continue
                center_x = round((xmin + xmax) / 2.0, 2)
                center_y = round((ymin + ymax) / 2.0, 2)
                foot_x = center_x
                foot_y = float(ymax)
                if polygon and not test_point_in_polygon(foot_x, foot_y, polygon):
                    continue
                if track_id_value is None:
                    track_id = synthetic_track_id
                    synthetic_track_id += 1
                else:
                    track_id = int(track_id_value)
                logical_frame_id = int(round((frame_id / max(actual_video_fps, 1e-9)) * target_timeline_fps))
                frame_key = f"{logical_frame_id:08d}"
                relative_sec = round(logical_frame_id / float(target_timeline_fps), 3)
                emitted_track_rows += 1
                track_rows.append(
                    {
                        "camera_id": camera_id,
                        "role": camera_cfg.get("role", ""),
                        "local_track_id": str(track_id),
                        "global_gt_id": int(track_id),
                        "source_track_surrogate_id": str(track_id),
                        "position_id": "",
                        "frame_id": logical_frame_id,
                        "source_frame_id_actual": frame_id,
                        "frame_key": frame_key,
                        "relative_sec": relative_sec,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "width": width,
                        "height": height,
                        "area": width * height,
                        "center_x": center_x,
                        "center_y": center_y,
                        "foot_x": foot_x,
                        "foot_y": foot_y,
                        "image_rel_path": f"VideoReplay\\{camera_id}\\{frame_key}.png",
                        "video_path": str(video_source),
                        "frame_source_mode": frame_source_mode,
                        "track_provider": "ultralytics_yolo_bytetrack",
                        "detection_score": round(float(det_score), 4),
                    }
                )
    finally:
        capture.release()

    track_rows.sort(key=lambda row: (row["frame_id"], row["global_gt_id"]))
    linking_started = time.perf_counter()
    track_rows, linking_runtime = link_short_gap_tracklets(
        track_rows,
        linking_cfg=linking_cfg,
    )
    linking_runtime["elapsed_sec"] = round(max(time.perf_counter() - linking_started, 1e-9), 6)
    elapsed_sec = max(time.perf_counter() - runtime_started, 1e-9)
    runtime_info = {
        "track_provider": "ultralytics_yolo_bytetrack",
        "detector_model": model_name,
        "tracker": tracker_name,
        "device": device,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "imgsz": imgsz,
        "person_class_id": person_class_id,
        "video_path": str(video_source),
        "start_frame": start_frame,
        "end_frame": end_frame if end_frame is not None else "",
        "frame_stride": frame_stride,
        "processed_frames": processed_frames,
        "raw_detection_count": raw_detection_count,
        "emitted_track_rows": emitted_track_rows,
        "elapsed_sec": round(elapsed_sec, 3),
        "processed_fps": round(processed_frames / elapsed_sec, 3),
        "actual_video_fps": round(actual_video_fps, 3),
        "target_timeline_fps": round(target_timeline_fps, 3),
        "actual_window_sec": round(((end_frame if end_frame is not None else frame_id) - start_frame) / max(actual_video_fps, 1e-9), 3),
        "identity_seed_field": "source_track_surrogate_id",
        "tracklet_linking": linking_runtime,
        "cache_runtime": cache_runtime,
        "notes": [
            "Track rows are generated from YOLO person detection plus ByteTrack on the real source video.",
            "ByteTrack output is passed through a short-gap intra-camera linker before the replay flow continues.",
            "global_gt_id is reused as a source-track surrogate ID in this inference-backed replay mode for compatibility with downstream audit files.",
            "GT-backed annotations are not used to create the source tracks in this mode.",
            "frame_id and relative_sec are normalized onto the configured timeline FPS so fake travel-time logic stays consistent with the replay topology.",
        ],
    }
    if cache_cfg.get("enabled", False):
        cache_write_runtime = write_inference_track_cache(cache_dir, signature, track_rows, runtime_info)
        runtime_info["cache_runtime"] = {
            "cache_enabled": True,
            "cache_hit": False,
            "cache_key": cache_write_runtime["cache_key"],
            "cache_root": cache_write_runtime["cache_root"],
            "metadata_json": cache_write_runtime["metadata_json"],
            "track_rows_json": cache_write_runtime["track_rows_json"],
            "runtime_json": cache_write_runtime["runtime_json"],
            "cache_created_at": cache_write_runtime["cache_created_at"],
            "cache_miss_reason": "refreshed" if cache_cfg.get("refresh_cache", False) else "cache_miss_computed",
            "cache_load_time_sec": 0.0,
            "time_saved_sec": 0.0,
        }
    return track_rows, runtime_info


def clone_track_rows_for_virtual_camera(source_rows, virtual_camera_id, virtual_role, time_offset_sec, frame_offset):
    cloned_rows = []
    for row in source_rows:
        cloned = copy.deepcopy(row)
        cloned["camera_id"] = virtual_camera_id
        cloned["role"] = virtual_role
        cloned["source_camera_id"] = row.get("camera_id", "")
        cloned["virtual_camera_id"] = virtual_camera_id
        cloned["replay_pass_index"] = ""
        cloned["fake_time_offset_sec"] = round(float(time_offset_sec), 3)
        cloned["fake_frame_offset"] = int(frame_offset)
        cloned["source_local_track_id"] = str(row.get("local_track_id", ""))
        cloned["local_track_id"] = f"{virtual_camera_id}_{row.get('local_track_id', '')}"
        cloned["frame_id"] = int(row["frame_id"]) + int(frame_offset)
        cloned["frame_key"] = f"{int(cloned['frame_id']):08d}"
        cloned["relative_sec"] = round(float(row["relative_sec"]) + float(time_offset_sec), 3)
        cloned_rows.append(cloned)
    cloned_rows.sort(key=lambda item: (item["frame_id"], item["global_gt_id"]))
    return cloned_rows


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


DEFAULT_FACE_BUFFER = {
    "enabled": True,
    "window_frames": 30,
    "max_frames": 8,
    "min_frame_step": 3,
}


def merged_face_buffer_cfg(config):
    merged = dict(DEFAULT_FACE_BUFFER)
    if isinstance(config, dict):
        merged.update({key: value for key, value in config.items() if value is not None})
    merged["window_frames"] = max(1, as_int(merged.get("window_frames", 30), 30))
    merged["max_frames"] = max(1, as_int(merged.get("max_frames", 8), 8))
    merged["min_frame_step"] = max(1, as_int(merged.get("min_frame_step", 3), 3))
    merged["enabled"] = bool(merged.get("enabled", True))
    return merged


def select_evidence_buffer_records(records, anchor_frame_id, preferred_record, buffer_cfg=None):
    cfg = merged_face_buffer_cfg(buffer_cfg)
    if not cfg["enabled"]:
        return []
    frame_min = max(0, as_int(anchor_frame_id))
    frame_max = frame_min + int(cfg["window_frames"])
    window_rows = []
    seen_frames = set()
    for row in records:
        frame_id = as_int(row.get("frame_id"))
        if frame_id < frame_min or frame_id > frame_max or frame_id in seen_frames:
            continue
        seen_frames.add(frame_id)
        window_rows.append(row)
    if not window_rows:
        window_rows = [preferred_record]
    selected = []
    last_frame_id = None
    preferred_frame_id = as_int(preferred_record.get("frame_id"))
    for row in sorted(window_rows, key=lambda item: (item["frame_id"], -item["area"])):
        frame_id = as_int(row.get("frame_id"))
        if last_frame_id is not None and frame_id - last_frame_id < int(cfg["min_frame_step"]) and frame_id != preferred_frame_id:
            continue
        selected.append(row)
        last_frame_id = frame_id
        if len(selected) >= int(cfg["max_frames"]):
            break
    if all(as_int(row.get("frame_id")) != preferred_frame_id for row in selected):
        selected.append(preferred_record)
    selected.sort(key=lambda item: (item["frame_id"], -item["area"]))
    return selected[: int(cfg["max_frames"])]


def crop_event_buffer_records(
    frame_cache: FrameSourceCache,
    dataset_root: Path,
    crop_root: Path,
    event_id,
    camera_id,
    records,
    head_cfg,
):
    buffer_rows = []
    camera_dir = crop_root / camera_id
    for index, record in enumerate(records, start=1):
        frame = frame_cache.frame_for_record(record, dataset_root)
        img_h, img_w = frame.shape[:2]
        body_rect = clamp_rect(
            {"x": record["xmin"], "y": record["ymin"], "width": record["width"], "height": record["height"]},
            img_w,
            img_h,
        )
        head_rect = get_head_rect(record, head_cfg, img_w, img_h)
        head_path = camera_dir / f"{event_id}_buffer_{index:02d}_{int(record['frame_id']):08d}_head.png"
        body_path = camera_dir / f"{event_id}_buffer_{index:02d}_{int(record['frame_id']):08d}_body.png"
        frame_cache.crop_record(record, head_rect, head_path, dataset_root)
        frame_cache.crop_record(record, body_rect, body_path, dataset_root)
        buffer_rows.append(
            {
                "frame_id": int(record["frame_id"]),
                "relative_sec": float(record["relative_sec"]),
                "bbox_area": int(record["area"]),
                "head_crop_path": str(head_path),
                "body_crop_path": str(body_path),
                "foot_x": float(record.get("foot_x", 0.0)),
                "foot_y": float(record.get("foot_y", 0.0)),
            }
        )
    return buffer_rows


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
    face_buffer_cfg = merged_face_buffer_cfg(wildtrack_config.get("face_buffer", {}))
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
            buffer_records = select_evidence_buffer_records(
                records,
                crossing_row["frame_id"],
                best_record,
                face_buffer_cfg,
            )
            evidence_buffer_rows = crop_event_buffer_records(
                frame_cache,
                dataset_root,
                crop_root,
                event_id,
                camera_id,
                buffer_records,
                head_cfg,
            )
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
                "evidence_buffer_count": len(evidence_buffer_rows),
                "evidence_buffer_json": json.dumps(evidence_buffer_rows, ensure_ascii=False),
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
                    "evidence_buffer_count": len(evidence_buffer_rows),
                    "evidence_buffer_json": json.dumps(evidence_buffer_rows, ensure_ascii=False),
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


def materialize_stage_inputs(
    project_root: Path,
    dataset_root: Path,
    wildtrack_config_path: Path,
    wildtrack_config,
    output_root: Path,
    track_rows_by_camera,
    frame_source_mode,
    low_load_cfg,
    transition_map,
    pipeline_backend="wildtrack_gt_annotations",
    extra_stage_summary=None,
):
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
        "pipeline_backend": pipeline_backend,
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
            "Direction filtering is applied before face matching and only ENTRY_IN candidates are exported to the identity queue.",
            "Frame crops are taken from the configured frame source mode for reproducible best-shot generation.",
        ],
    }
    if pipeline_backend == "wildtrack_gt_annotations":
        stage_summary["notes"].insert(
            0,
            "This offline stage uses GT-backed track rows as the detect/track provider for the thesis demo baseline.",
        )
    else:
        stage_summary["notes"].insert(
            0,
            "This offline stage uses replayed source track rows from the configured provider instead of GT-backed track generation.",
        )
    if extra_stage_summary:
        stage_summary.update(copy.deepcopy(extra_stage_summary))
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

    source_backend = offline_config.get("source_backend", "wildtrack_gt_annotations")
    if source_backend == "single_source_sequential_replay":
        replay_cfg = offline_config.get("single_source_replay", {}) or {}
        track_provider = replay_cfg.get("track_provider", "inference_yolo_bytetrack")
        source_template_path = Path(replay_cfg.get("source_template_config", ""))
        if not source_template_path.is_absolute():
            source_template_path = (project_root / source_template_path).resolve()
        if not source_template_path.exists():
            raise RuntimeError(f"Missing single_source_replay.source_template_config: {source_template_path}")
        source_template = load_json(source_template_path)
        source_camera_id = replay_cfg.get("source_camera_id", "")
        if not source_camera_id:
            raise RuntimeError("single_source_replay.source_camera_id is required.")
        if source_camera_id not in source_template.get("cameras", {}):
            raise RuntimeError(f"Source camera {source_camera_id} not found in {source_template_path}")
        virtual_camera_ids = replay_cfg.get("virtual_camera_ids", []) or []
        if not virtual_camera_ids:
            raise RuntimeError("single_source_replay.virtual_camera_ids is required.")
        if virtual_camera_ids != wildtrack_config.get("selected_cameras", []):
            raise RuntimeError(
                "single_source_replay.virtual_camera_ids must match wildtrack_config.selected_cameras for the sequential replay demo."
            )
        time_offsets = replay_cfg.get("virtual_time_offsets_sec", []) or []
        if not time_offsets:
            raise RuntimeError("single_source_replay.virtual_time_offsets_sec is required.")
        if len(time_offsets) != len(virtual_camera_ids):
            raise RuntimeError("single_source_replay.virtual_time_offsets_sec length must match virtual_camera_ids length.")
        fps = float(wildtrack_config["assumed_video_fps"])
        configured_frame_offsets = replay_cfg.get("virtual_frame_offsets", []) or []
        if configured_frame_offsets and len(configured_frame_offsets) != len(virtual_camera_ids):
            raise RuntimeError("single_source_replay.virtual_frame_offsets length must match virtual_camera_ids length.")
        frame_offsets = (
            [int(item) for item in configured_frame_offsets]
            if configured_frame_offsets
            else [int(round(float(item) * fps)) for item in time_offsets]
        )
        source_video_path = Path(replay_cfg.get("source_video", ""))
        if not source_video_path.is_absolute():
            source_video_path = (project_root / source_video_path).resolve()
        if not source_video_path.exists():
            raise RuntimeError(f"Missing single-source replay video: {source_video_path}")
        frame_source_mode = offline_config.get("frame_source_mode", "video_files")
        low_load_cfg = offline_config.get("low_load", {})
        source_camera_cfg = copy.deepcopy(source_template["cameras"][source_camera_id])
        first_virtual_camera = wildtrack_config["cameras"][virtual_camera_ids[0]]
        if first_virtual_camera.get("processing_roi"):
            source_camera_cfg["processing_roi"] = copy.deepcopy(first_virtual_camera.get("processing_roi", []))
            source_camera_cfg["entry_roi"] = copy.deepcopy(first_virtual_camera.get("processing_roi", []))
        source_track_runtime = {}
        if track_provider == "gt_annotations":
            source_rows = extract_camera_track_rows(
                dataset_root,
                source_camera_id,
                source_camera_cfg,
                fps,
                source_video_path,
                low_load_cfg,
                frame_source_mode,
            )
            source_track_runtime = {
                "track_provider": "gt_annotations",
                "video_path": str(source_video_path),
                "start_frame": as_int(low_load_cfg.get("start_frame", 0)),
                "end_frame": low_load_cfg.get("end_frame", ""),
                "frame_stride": max(1, as_int(low_load_cfg.get("frame_stride", 1), 1)),
                "identity_seed_field": "global_gt_id",
                "notes": [
                    "This legacy mode seeds track rows directly from Wildtrack annotations.",
                    "It is kept only as an explicit opt-in fallback for audit/regression, not as the default runnable path.",
                ],
            }
        elif track_provider == "inference_yolo_bytetrack":
            inference_cfg = copy.deepcopy(replay_cfg.get("inference", {}) or {})
            cache_cfg = (inference_cfg.get("cache", {}) or {})
            if cache_cfg.get("cache_dir"):
                cache_dir = Path(cache_cfg["cache_dir"])
                if not cache_dir.is_absolute():
                    cache_cfg["cache_dir"] = str((project_root / cache_dir).resolve())
                inference_cfg["cache"] = cache_cfg
            source_rows, source_track_runtime = extract_inference_track_rows(
                source_camera_id,
                source_camera_cfg,
                fps,
                source_video_path,
                low_load_cfg,
                frame_source_mode,
                inference_cfg,
            )
        else:
            raise RuntimeError(
                f"Unsupported single_source_replay.track_provider={track_provider!r}. "
                "Expected 'inference_yolo_bytetrack' or explicit legacy opt-in 'gt_annotations'."
            )
        track_rows_by_camera = {}
        replay_manifest_rows = []
        for index, virtual_camera_id in enumerate(virtual_camera_ids):
            cloned_rows = clone_track_rows_for_virtual_camera(
                source_rows,
                virtual_camera_id,
                wildtrack_config["cameras"][virtual_camera_id].get("role", "entry"),
                float(time_offsets[index]),
                int(frame_offsets[index]),
            )
            for row in cloned_rows:
                row["replay_pass_index"] = index + 1
            track_rows_by_camera[virtual_camera_id] = cloned_rows
            replay_manifest_rows.append(
                {
                    "virtual_camera_id": virtual_camera_id,
                    "source_camera_id": source_camera_id,
                    "source_video": str(source_video_path),
                    "replay_pass_index": index + 1,
                    "fake_time_offset_sec": round(float(time_offsets[index]), 3),
                    "fake_frame_offset": int(frame_offsets[index]),
                    "role": wildtrack_config["cameras"][virtual_camera_id].get("role", "entry"),
                }
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
            pipeline_backend="single_source_sequential_replay",
            extra_stage_summary={
                "single_source_replay": {
                    "source_camera_id": source_camera_id,
                    "track_provider": track_provider,
                    "source_template_config": str(source_template_path),
                    "source_video": str(source_video_path),
                    "virtual_camera_ids": virtual_camera_ids,
                    "virtual_time_offsets_sec": [round(float(item), 3) for item in time_offsets],
                    "virtual_frame_offsets": [int(item) for item in frame_offsets],
                    "source_track_runtime": source_track_runtime,
                },
                "notes": [
                    "This run replays one real source video sequentially as multiple virtual cameras for the supervisor-approved sanity demo.",
                    "All virtual cameras reuse the same source track rows with fake time offsets to test Unknown_Global_ID reuse in the easiest setting.",
                    "The goal of this mode is to validate sequential cross-camera unknown reuse before returning to harder real multi-camera overlap scenarios.",
                ],
            },
        )
        save_json(output_root / "summaries" / "single_source_replay_manifest.json", replay_manifest_rows)
        result["video_sources"] = {camera_id: str(source_video_path) for camera_id in virtual_camera_ids}
        result["replay_manifest_json"] = str(output_root / "summaries" / "single_source_replay_manifest.json")
        result["source_backend"] = "single_source_sequential_replay"
        return result

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
