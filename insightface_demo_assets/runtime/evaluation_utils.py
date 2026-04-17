import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def ensure_numpy_asfarray_compat():
    # motmetrics 1.4.0 still calls np.asfarray, which was removed in NumPy 2.x.
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda values, dtype=float: np.asarray(values, dtype=dtype)


def safe_float(value, default=0.0):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        fieldnames = []
    elif not fieldnames:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_identity_id(row):
    return row.get("resolved_global_id") or row.get("unknown_global_id") or row.get("matched_known_id") or ""


def _event_anchor_observation(row):
    buffer_json = row.get("evidence_buffer_json", "")
    if buffer_json:
        try:
            items = json.loads(buffer_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            items = []
        if items:
            first_item = items[0]
            return {
                "frame_id": safe_int(first_item.get("frame_id"), safe_int(row.get("frame_id"))),
                "foot_x": safe_float(first_item.get("foot_x"), safe_float(row.get("foot_x"))),
                "foot_y": safe_float(first_item.get("foot_y"), safe_float(row.get("foot_y"))),
            }
    return {
        "frame_id": safe_int(row.get("frame_id")),
        "foot_x": safe_float(row.get("foot_x")),
        "foot_y": safe_float(row.get("foot_y")),
    }


def bbox_xyxy(row, prefix=""):
    return (
        safe_float(row[f"{prefix}xmin"]),
        safe_float(row[f"{prefix}ymin"]),
        safe_float(row[f"{prefix}xmax"]),
        safe_float(row[f"{prefix}ymax"]),
    )


def bbox_xywh(row, prefix=""):
    xmin, ymin, xmax, ymax = bbox_xyxy(row, prefix=prefix)
    return [xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin)]


def bbox_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _event_match_cost(gt_row, pred_row, frame_tolerance, max_foot_distance_px):
    gt_anchor = _event_anchor_observation(gt_row)
    pred_anchor = _event_anchor_observation(pred_row)
    frame_gap = abs(gt_anchor["frame_id"] - pred_anchor["frame_id"])
    if frame_gap > frame_tolerance:
        return math.inf
    gt_x = gt_anchor["foot_x"]
    gt_y = gt_anchor["foot_y"]
    pred_x = pred_anchor["foot_x"]
    pred_y = pred_anchor["foot_y"]
    foot_distance = ((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2) ** 0.5
    if foot_distance > max_foot_distance_px:
        return math.inf
    # Smaller frame gap dominates first, then smaller foot distance breaks ties.
    return frame_gap * 1000.0 + foot_distance


def match_event_rows_to_gt(gt_rows, pred_rows, frame_tolerance=3, max_foot_distance_px=80.0):
    matches = []
    unmatched_gt = []
    unmatched_pred = []
    gt_by_camera = defaultdict(list)
    pred_by_camera = defaultdict(list)
    for row in gt_rows:
        gt_by_camera[row["camera_id"]].append(row)
    for row in pred_rows:
        pred_by_camera[row["camera_id"]].append(row)

    for camera_id in sorted(set(gt_by_camera) | set(pred_by_camera)):
        camera_gt = gt_by_camera.get(camera_id, [])
        camera_pred = pred_by_camera.get(camera_id, [])
        if not camera_gt:
            unmatched_pred.extend(camera_pred)
            continue
        if not camera_pred:
            unmatched_gt.extend(camera_gt)
            continue

        cost_matrix = np.full((len(camera_gt), len(camera_pred)), fill_value=1e9, dtype=np.float32)
        for gt_index, gt_row in enumerate(camera_gt):
            for pred_index, pred_row in enumerate(camera_pred):
                cost = _event_match_cost(gt_row, pred_row, frame_tolerance, max_foot_distance_px)
                if math.isfinite(cost):
                    cost_matrix[gt_index, pred_index] = cost

        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        matched_gt = set()
        matched_pred = set()
        for gt_index, pred_index in zip(gt_indices, pred_indices):
            if cost_matrix[gt_index, pred_index] >= 1e9:
                continue
            gt_row = camera_gt[gt_index]
            pred_row = camera_pred[pred_index]
            gt_anchor = _event_anchor_observation(gt_row)
            pred_anchor = _event_anchor_observation(pred_row)
            frame_gap = abs(gt_anchor["frame_id"] - pred_anchor["frame_id"])
            foot_distance = ((gt_anchor["foot_x"] - pred_anchor["foot_x"]) ** 2 + (gt_anchor["foot_y"] - pred_anchor["foot_y"]) ** 2) ** 0.5
            matches.append(
                {
                    "camera_id": camera_id,
                    "gt_event_id": gt_row.get("event_id", ""),
                    "pred_event_id": pred_row.get("event_id", ""),
                    "gt_global_id": str(gt_row.get("global_gt_id", "")),
                    "pred_identity_id": normalize_identity_id(pred_row),
                    "gt_frame_id": gt_anchor["frame_id"],
                    "pred_frame_id": pred_anchor["frame_id"],
                    "frame_gap": frame_gap,
                    "foot_distance_px": round(foot_distance, 4),
                }
            )
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)

        unmatched_gt.extend(camera_gt[index] for index in range(len(camera_gt)) if index not in matched_gt)
        unmatched_pred.extend(camera_pred[index] for index in range(len(camera_pred)) if index not in matched_pred)

    return matches, unmatched_gt, unmatched_pred


def compute_event_level_idf1(gt_rows, pred_rows, matches):
    pair_counts = Counter()
    gt_identity_set = set()
    pred_identity_set = set()
    for row in gt_rows:
        gt_identity_set.add(str(row.get("global_gt_id", "")))
    for row in pred_rows:
        identity_id = normalize_identity_id(row)
        if identity_id:
            pred_identity_set.add(identity_id)
    for row in matches:
        gt_id = row["gt_global_id"]
        pred_id = row["pred_identity_id"]
        if not gt_id or not pred_id:
            continue
        pair_counts[(gt_id, pred_id)] += 1

    gt_ids = sorted({gt_id for gt_id, _pred_id in pair_counts})
    pred_ids = sorted({pred_id for _gt_id, pred_id in pair_counts})
    if gt_ids and pred_ids:
        score_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
        for gt_index, gt_id in enumerate(gt_ids):
            for pred_index, pred_id in enumerate(pred_ids):
                score_matrix[gt_index, pred_index] = pair_counts[(gt_id, pred_id)]
        gt_indices, pred_indices = linear_sum_assignment(-score_matrix)
        idtp = int(score_matrix[gt_indices, pred_indices].sum())
    else:
        idtp = 0

    total_gt = len(gt_rows)
    total_pred = len(pred_rows)
    idfn = max(0, total_gt - idtp)
    idfp = max(0, total_pred - idtp)
    id_precision = idtp / max(idtp + idfp, 1)
    id_recall = idtp / max(idtp + idfn, 1)
    idf1 = (2 * idtp) / max((2 * idtp) + idfp + idfn, 1)

    return {
        "gt_event_count": total_gt,
        "pred_event_count": total_pred,
        "matched_event_count": len(matches),
        "gt_identity_count": len(gt_identity_set),
        "pred_identity_count": len(pred_identity_set),
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "id_precision": round(id_precision, 4),
        "id_recall": round(id_recall, 4),
        "idf1": round(idf1, 4),
    }


def compute_local_tracking_mota(gt_rows, pred_rows, iou_threshold=0.5):
    ensure_numpy_asfarray_compat()
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    frames = sorted({safe_int(row.get("frame_id")) for row in gt_rows} | {safe_int(row.get("frame_id")) for row in pred_rows})
    gt_by_frame = defaultdict(list)
    pred_by_frame = defaultdict(list)
    for row in gt_rows:
        gt_by_frame[safe_int(row.get("frame_id"))].append(row)
    for row in pred_rows:
        pred_by_frame[safe_int(row.get("frame_id"))].append(row)

    # motmetrics on the current pandas/numpy stack still tries to cast object IDs to
    # float internally. We keep the original string IDs in the CSV artifacts, but remap
    # them to stable integer surrogates here so MOTA/IDF1 can be computed reliably.
    gt_id_map = {}
    pred_id_map = {}

    def mapped_id(raw_id, id_map):
        raw_id = str(raw_id)
        if raw_id not in id_map:
            id_map[raw_id] = len(id_map) + 1
        return id_map[raw_id]

    for frame_id in frames:
        gt_frame_rows = gt_by_frame.get(frame_id, [])
        pred_frame_rows = pred_by_frame.get(frame_id, [])
        gt_ids = [mapped_id(row.get("global_gt_id"), gt_id_map) for row in gt_frame_rows]
        pred_ids = [
            mapped_id(
                row.get("local_track_id") or row.get("source_track_surrogate_id") or row.get("global_gt_id"),
                pred_id_map,
            )
            for row in pred_frame_rows
        ]
        gt_boxes = np.asarray([bbox_xywh(row) for row in gt_frame_rows], dtype=float) if gt_frame_rows else np.empty((0, 4))
        pred_boxes = np.asarray([bbox_xywh(row) for row in pred_frame_rows], dtype=float) if pred_frame_rows else np.empty((0, 4))
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=(1.0 - float(iou_threshold)))
        acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "mota",
            "idf1",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "num_objects",
            "precision",
            "recall",
        ],
        name="local_tracking",
    )
    row = summary.loc["local_tracking"]
    return {
        "frame_count": int(row["num_frames"]),
        "gt_detection_count": int(row["num_objects"]),
        "pred_track_count": len({str(item.get("local_track_id") or item.get("source_track_surrogate_id") or item.get("global_gt_id")) for item in pred_rows}),
        "gt_track_count": len({str(item.get("global_gt_id")) for item in gt_rows}),
        "mota": round(float(row["mota"]), 4),
        "tracking_idf1": round(float(row["idf1"]), 4),
        "id_switches": int(row["num_switches"]),
        "fp": int(row["num_false_positives"]),
        "fn": int(row["num_misses"]),
        "precision": round(float(row["precision"]), 4),
        "recall": round(float(row["recall"]), 4),
    }


def aggregate_local_tracking_metrics(per_camera_metrics):
    if not per_camera_metrics:
        return {
            "frame_count": 0,
            "gt_detection_count": 0,
            "pred_track_count": 0,
            "gt_track_count": 0,
            "mota": 0.0,
            "tracking_idf1": 0.0,
            "id_switches": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "per_camera": {},
        }
    total_gt = sum(int(item.get("gt_detection_count", 0)) for item in per_camera_metrics.values())
    total_fp = sum(int(item.get("fp", 0)) for item in per_camera_metrics.values())
    total_fn = sum(int(item.get("fn", 0)) for item in per_camera_metrics.values())
    total_switches = sum(int(item.get("id_switches", 0)) for item in per_camera_metrics.values())
    total_frames = sum(int(item.get("frame_count", 0)) for item in per_camera_metrics.values())
    total_pred_tracks = sum(int(item.get("pred_track_count", 0)) for item in per_camera_metrics.values())
    total_gt_tracks = sum(int(item.get("gt_track_count", 0)) for item in per_camera_metrics.values())
    total_tp = max(0, total_gt - total_fn)
    mota = 1.0 - ((total_fp + total_fn + total_switches) / max(total_gt, 1))
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    weighted_idf1 = 0.0
    if total_gt > 0:
        weighted_idf1 = sum(
            float(item.get("tracking_idf1", 0.0)) * int(item.get("gt_detection_count", 0))
            for item in per_camera_metrics.values()
        ) / float(total_gt)
    return {
        "frame_count": total_frames,
        "gt_detection_count": total_gt,
        "pred_track_count": total_pred_tracks,
        "gt_track_count": total_gt_tracks,
        "mota": round(float(mota), 4),
        "tracking_idf1": round(float(weighted_idf1), 4),
        "id_switches": total_switches,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "per_camera": per_camera_metrics,
    }


def build_unknown_timeline(resolved_rows):
    grouped = defaultdict(list)
    for row in resolved_rows:
        identity_status = row.get("identity_status", "")
        resolved_id = normalize_identity_id(row)
        if identity_status != "unknown" or not resolved_id:
            continue
        grouped[resolved_id].append(row)

    timeline_rows = []
    timeline_json = []
    for identity_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(rows, key=lambda row: (safe_float(row.get("relative_sec")), row.get("camera_id", ""), row.get("event_id", "")))
        appearances = []
        representative_snapshot = ""
        representative_head = ""
        for row in ordered:
            snapshot_path = row.get("best_body_crop", "")
            head_snapshot_path = row.get("best_head_crop", "")
            if not representative_snapshot:
                representative_snapshot = snapshot_path
                representative_head = head_snapshot_path
            appearance = {
                "event_id": row.get("event_id", ""),
                "camera_id": row.get("camera_id", ""),
                "relative_sec": round(safe_float(row.get("relative_sec")), 3),
                "relation_type": row.get("relation_type", ""),
                "zone_id": row.get("zone_id", ""),
                "subzone_id": row.get("subzone_id", ""),
                "identity_status": row.get("identity_status", ""),
                "ui_identity_state": row.get("ui_identity_state", ""),
                "ui_identity_label": row.get("ui_identity_label", ""),
                "modality_primary_used": row.get("modality_primary_used", ""),
                "modality_secondary_used": row.get("modality_secondary_used", ""),
                "decision_reason": row.get("decision_reason", ""),
                "reason_code": row.get("reason_code", ""),
                "best_body_crop": snapshot_path,
                "best_head_crop": head_snapshot_path,
            }
            appearances.append(appearance)
            timeline_rows.append(
                {
                    "identity_id": identity_id,
                    "appearance_index": len(appearances),
                    **appearance,
                }
            )
        timeline_json.append(
            {
                "identity_id": identity_id,
                "identity_label": ordered[0].get("ui_identity_label") or identity_id,
                "identity_status": "unknown",
                "appearance_count": len(appearances),
                "camera_sequence": [item["camera_id"] for item in appearances],
                "first_seen_camera": appearances[0]["camera_id"],
                "first_seen_relative_sec": appearances[0]["relative_sec"],
                "last_seen_camera": appearances[-1]["camera_id"],
                "last_seen_relative_sec": appearances[-1]["relative_sec"],
                "representative_snapshot_path": representative_snapshot,
                "representative_head_snapshot_path": representative_head,
                "appearances": appearances,
            }
        )
    return timeline_rows, timeline_json


def summarize_unknown_handoffs(resolved_rows):
    grouped = defaultdict(list)
    for row in resolved_rows:
        identity_id = normalize_identity_id(row)
        if row.get("identity_status") != "unknown" or not identity_id:
            continue
        grouped[identity_id].append(row)
    edge_counts = Counter()
    sequences = []
    for identity_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(rows, key=lambda row: (safe_float(row.get("relative_sec")), row.get("camera_id", ""), row.get("event_id", "")))
        camera_sequence = []
        for row in ordered:
            camera_id = row.get("camera_id", "")
            if not camera_sequence or camera_sequence[-1] != camera_id:
                camera_sequence.append(camera_id)
        for index in range(1, len(camera_sequence)):
            edge_counts[(camera_sequence[index - 1], camera_sequence[index])] += 1
        sequences.append(
            {
                "identity_id": identity_id,
                "camera_sequence": camera_sequence,
                "appearance_count": len(ordered),
                "first_seen_relative_sec": round(safe_float(ordered[0].get("relative_sec")), 3) if ordered else 0.0,
                "last_seen_relative_sec": round(safe_float(ordered[-1].get("relative_sec")), 3) if ordered else 0.0,
            }
        )
    multi_camera_sequences = [item for item in sequences if len(item["camera_sequence"]) >= 2]
    return {
        "unknown_identity_count": len(sequences),
        "multi_camera_identity_count": len(multi_camera_sequences),
        "handoff_edge_count": sum(edge_counts.values()),
        "handoff_edges": [
            {"src_camera_id": src, "dst_camera_id": dst, "count": count}
            for (src, dst), count in sorted(edge_counts.items())
        ],
        "identity_sequences": sequences,
    }


def summarize_latency_records(records):
    if not records:
        return {
            "count": 0,
            "avg_latency_sec": 0.0,
            "p50_latency_sec": 0.0,
            "p95_latency_sec": 0.0,
            "max_latency_sec": 0.0,
        }
    latencies = np.asarray(records, dtype=np.float32)
    return {
        "count": int(latencies.size),
        "avg_latency_sec": round(float(latencies.mean()), 4),
        "p50_latency_sec": round(float(np.percentile(latencies, 50)), 4),
        "p95_latency_sec": round(float(np.percentile(latencies, 95)), 4),
        "max_latency_sec": round(float(latencies.max()), 4),
    }


def nearest_available_annotation_frame(annotation_dir: Path, actual_frame_id):
    annotation_dir = Path(annotation_dir)
    candidates = [path for path in annotation_dir.glob("*.json")]
    target = safe_int(actual_frame_id)
    if not candidates:
        raise FileNotFoundError(f"No annotation JSON files found in {annotation_dir}")
    best_path = min(candidates, key=lambda path: abs(safe_int(path.stem) - target))
    return best_path, abs(safe_int(best_path.stem) - target)


def _point_in_polygon(x, y, polygon):
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


def extract_aligned_gt_rows_from_annotations(annotation_dir: Path, view_index, logical_to_actual_frames, processing_polygon=None):
    rows = []
    annotation_cache = {}
    for logical_frame_id, actual_frame_id in sorted(logical_to_actual_frames.items()):
        annotation_path, frame_gap = nearest_available_annotation_frame(annotation_dir, actual_frame_id)
        payload = annotation_cache.get(annotation_path)
        if payload is None:
            payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
            annotation_cache[annotation_path] = payload
        for obj in payload:
            view = next((item for item in obj.get("views", []) if safe_int(item.get("viewNum")) == safe_int(view_index)), None)
            if not view:
                continue
            xmin = safe_float(view.get("xmin"), -1.0)
            ymin = safe_float(view.get("ymin"), -1.0)
            xmax = safe_float(view.get("xmax"), -1.0)
            ymax = safe_float(view.get("ymax"), -1.0)
            if xmin < 0.0 or ymin < 0.0 or xmax <= xmin or ymax <= ymin:
                continue
            foot_x = round((xmin + xmax) / 2.0, 2)
            foot_y = ymax
            if processing_polygon and not _point_in_polygon(foot_x, foot_y, processing_polygon):
                continue
            rows.append(
                {
                    "frame_id": safe_int(logical_frame_id),
                    "actual_frame_id": safe_int(actual_frame_id),
                    "matched_annotation_frame_id": safe_int(annotation_path.stem),
                    "annotation_frame_gap": frame_gap,
                    "global_gt_id": safe_int(obj.get("personID")),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                }
            )
    return rows
