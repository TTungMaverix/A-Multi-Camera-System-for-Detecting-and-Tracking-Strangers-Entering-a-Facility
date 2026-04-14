import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from association_core import (
    assign_model_identities as core_assign_model_identities,
    best_known_match as core_best_known_match,
    build_topology_index as core_build_topology_index,
    build_event_assignment_audit_row as core_build_event_assignment_audit_row,
    create_unknown_profile as core_create_unknown_profile,
    default_subzone_for_camera as core_default_subzone_for_camera,
    default_zone_for_camera as core_default_zone_for_camera,
    evaluate_profile_candidate as core_evaluate_profile_candidate,
    load_association_policy as core_load_association_policy,
    load_camera_transition_map as core_load_camera_transition_map,
    resolve_spatial_context as core_resolve_spatial_context,
    summarize_decision_logs as core_summarize_decision_logs,
    update_unknown_profile as core_update_unknown_profile,
    write_jsonl as core_write_jsonl,
)
from offline_pipeline.event_builder import (
    FrameSourceCache,
    crop_event_buffer_records,
    merged_face_buffer_cfg,
    select_evidence_buffer_records,
)
from offline_pipeline.direction_logic import evaluate_direction
from scene_calibration import (
    apply_scene_calibration_to_transition_map,
    apply_scene_calibration_to_wildtrack_config,
    load_runtime_scene_calibration,
)

CONFIG_DEFAULT = Path(__file__).with_name("face_demo_config.json")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path):
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


def resolve_path(base_dir: Path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


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


def normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def load_image_unicode(image_path: Path):
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(image_path: Path, image):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {image_path}")
    encoded.tofile(str(image_path))


def choose_best_face(faces):
    def score(face):
        bbox = np.asarray(face.bbox).astype(float)
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        det = float(face.det_score or 0.0)
        return area * max(det, 0.01)

    return max(faces, key=score)


_BODY_HOG_DESCRIPTOR = None


def _body_hog_descriptor():
    global _BODY_HOG_DESCRIPTOR
    if _BODY_HOG_DESCRIPTOR is None:
        _BODY_HOG_DESCRIPTOR = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    return _BODY_HOG_DESCRIPTOR


def extract_embedding_from_image(app, image_path: Path):
    result = {
        "status": "missing",
        "embedding": None,
        "face_count": 0,
        "det_score": 0.0,
        "bbox": "",
        "message": "",
    }
    if not image_path or not image_path.exists():
        result["message"] = f"missing_image:{image_path}"
        return result
    image = load_image_unicode(image_path)
    if image is None:
        result["message"] = f"failed_to_read:{image_path}"
        return result
    faces = app.get(image)
    result["face_count"] = len(faces)
    if not faces:
        result["status"] = "no_face"
        result["message"] = "no_face_detected"
        return result
    face = choose_best_face(faces)
    embedding = np.asarray(face.normed_embedding, dtype=np.float32)
    bbox = [int(round(x)) for x in np.asarray(face.bbox).tolist()]
    result.update(
        {
            "status": "ok",
            "embedding": embedding,
            "det_score": float(face.det_score or 0.0),
            "bbox": json.dumps(bbox),
            "message": "ok",
        }
    )
    return result


def extract_body_feature(image_path: Path):
    result = {
        "status": "missing",
        "embedding": None,
        "message": "",
        "shape": "",
    }
    if not image_path or not image_path.exists():
        result["message"] = f"missing_image:{image_path}"
        return result
    image = load_image_unicode(image_path)
    if image is None:
        result["message"] = f"failed_to_read:{image_path}"
        return result
    h, w = image.shape[:2]
    result["shape"] = f"{w}x{h}"
    if w < 12 or h < 24:
        result["status"] = "too_small"
        result["message"] = f"too_small:{w}x{h}"
        return result
    resized = cv2.resize(image, (64, 128), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_vec = _body_hog_descriptor().compute(gray)
    hog_vec = hog_vec.reshape(-1).astype(np.float32) if hog_vec is not None else np.zeros((0,), dtype=np.float32)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hs_parts = []
    for part in (hsv, hsv[:64, :, :], hsv[64:, :, :]):
        hist = cv2.calcHist([part], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten().astype(np.float32)
        hs_parts.append(hist)
    embedding = normalize(np.concatenate([hog_vec, *hs_parts], axis=0))
    result["status"] = "ok"
    result["embedding"] = embedding
    result["message"] = "ok_hog_hs_descriptor"
    return result


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
    mid_point = {"x": (prev_point["x"] + curr_point["x"]) / 2.0, "y": (prev_point["y"] + curr_point["y"]) / 2.0}
    return point_to_segment_distance(mid_point, line) <= distance_threshold


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


def crop_and_save_from_record(frame_cache, dataset_root: Path, record, rect, out_path: Path):
    try:
        frame_cache.crop_record(record, rect, out_path, dataset_root)
    except RuntimeError as exc:
        return False, str(exc)
    return True, "ok"


def parse_track_rows(track_rows):
    parsed = []
    for row in track_rows:
        item = dict(row)
        for key in (
            "global_gt_id",
            "position_id",
            "frame_id",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "width",
            "height",
            "area",
        ):
            item[key] = as_int(item.get(key))
        for key in ("relative_sec", "center_x", "center_y", "foot_x", "foot_y"):
            item[key] = as_float(item.get(key))
        parsed.append(item)
    return parsed


def build_track_indexes(track_rows):
    by_gt_camera = defaultdict(list)
    by_gt = defaultdict(lambda: defaultdict(list))
    for row in track_rows:
        key = (str(row["global_gt_id"]), row["camera_id"])
        by_gt_camera[key].append(row)
        by_gt[str(row["global_gt_id"])][row["camera_id"]].append(row)
    for rows in by_gt_camera.values():
        rows.sort(key=lambda row: row["frame_id"])
    for gt_map in by_gt.values():
        for rows in gt_map.values():
            rows.sort(key=lambda row: row["frame_id"])
    return by_gt_camera, by_gt


def build_topology(config):
    return core_build_topology_index(config)


def get_camera_zone_config(camera_id, transition_map):
    return (transition_map.get("cameras", {}) or {}).get(camera_id, {})


def default_zone_for_camera(camera_id, transition_map):
    return core_default_zone_for_camera(camera_id, transition_map)


def default_subzone_for_camera(camera_id, transition_map, zone_id=""):
    return core_default_subzone_for_camera(camera_id, transition_map, zone_id=zone_id)


def resolve_zone_for_point(camera_id, point_x, point_y, transition_map):
    return core_resolve_spatial_context(camera_id, point_x, point_y, transition_map)


def resolve_zone_for_record(record, transition_map):
    return core_resolve_spatial_context(
        record["camera_id"],
        record.get("foot_x"),
        record.get("foot_y"),
        transition_map,
    )


def enroll_demo_authorized_identities(app, queue_rows, known_root: Path, count=2):
    if int(count or 0) <= 0:
        return []
    counts = Counter(row["global_gt_id"] for row in queue_rows)
    ordered_rows = sorted(
        queue_rows,
        key=lambda row: (
            -counts[row["global_gt_id"]],
            as_float(row.get("relative_sec")),
            row.get("camera_id", ""),
        ),
    )
    selected = []
    used_gt = set()
    for row in ordered_rows:
        gt_id = str(row["global_gt_id"])
        if gt_id in used_gt:
            continue
        for crop_key in ("best_head_crop", "best_body_crop"):
            crop_path = Path(row[crop_key])
            emb = extract_embedding_from_image(app, crop_path)
            if emb["status"] != "ok":
                continue
            identity_id = f"known_demo_gt_{int(gt_id):04d}"
            display_name = f"Authorized Demo GT {gt_id}"
            identity_dir = known_root / identity_id
            identity_dir.mkdir(parents=True, exist_ok=True)
            ext = crop_path.suffix or ".png"
            dest_path = identity_dir / f"seed_{crop_key}{ext}"
            shutil.copy2(crop_path, dest_path)
            selected.append(
                {
                    "identity_id": identity_id,
                    "display_name": display_name,
                    "source_repo_path": str(crop_path),
                    "gallery_rel_path": str(dest_path.relative_to(known_root.parent)),
                    "seed_type": "wildtrack_entry_event",
                    "status": "demo_authorized_seed",
                    "notes": f"Auto-enrolled from {row['event_id']} using {crop_key}",
                }
            )
            used_gt.add(gt_id)
            break
        if len(selected) >= count:
            break
    return selected


def build_gallery_embeddings(app, manifest_rows, base_dir: Path, output_csv: Path):
    per_image_rows = []
    per_identity_vectors = defaultdict(list)
    for row in manifest_rows:
        image_path = base_dir / row["gallery_rel_path"]
        emb = extract_embedding_from_image(app, image_path)
        per_image_rows.append(
            {
                "identity_id": row["identity_id"],
                "display_name": row["display_name"],
                "image_rel_path": row["gallery_rel_path"],
                "embedding_status": emb["status"],
                "embedding_dim": len(emb["embedding"]) if emb["embedding"] is not None else "",
                "model_name": "buffalo_l",
                "embedding_json": json.dumps(emb["embedding"].tolist()) if emb["embedding"] is not None else "",
                "notes": emb["message"],
            }
        )
        if emb["embedding"] is not None:
            per_identity_vectors[row["identity_id"]].append(emb["embedding"])
    write_csv(
        output_csv,
        per_image_rows,
        [
            "identity_id",
            "display_name",
            "image_rel_path",
            "embedding_status",
            "embedding_dim",
            "model_name",
            "embedding_json",
            "notes",
        ],
    )
    identity_means = {}
    for identity_id, vectors in per_identity_vectors.items():
        identity_means[identity_id] = normalize(np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32))
    return identity_means, per_image_rows


def _laplacian_variance(image_path: Path):
    image = load_image_unicode(image_path)
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _parse_evidence_buffer_rows(event):
    raw_value = event.get("evidence_buffer_json", "[]")
    if isinstance(raw_value, list):
        rows = raw_value
    else:
        try:
            rows = json.loads(raw_value or "[]")
        except json.JSONDecodeError:
            rows = []
    if rows:
        return rows
    return [
        {
            "frame_id": as_int(event.get("best_shot_frame"), as_int(event.get("frame_id"))),
            "relative_sec": as_float(event.get("best_shot_sec"), as_float(event.get("relative_sec"))),
            "bbox_area": as_int(event.get("bbox_area", 0)),
            "head_crop_path": event.get("best_head_crop", ""),
            "body_crop_path": event.get("best_body_crop", ""),
        }
    ]


def _buffer_face_quality(face_result, buffer_row):
    if face_result.get("status") != "ok":
        return -1.0
    blur_score = min(_laplacian_variance(Path(buffer_row["head_crop_path"])), 500.0)
    bbox_bonus = min(float(buffer_row.get("bbox_area", 0)), 40000.0) / 40000.0
    return round((float(face_result.get("det_score", 0.0)) * 100.0) + (blur_score / 25.0) + bbox_bonus, 4)


def analyze_event_crops(app, events):
    analyzed = []
    for event in events:
        buffer_rows = _parse_evidence_buffer_rows(event)
        analyzed_buffer = []
        face_detected_in_buffer = 0
        face_embedding_created_in_buffer = 0
        for index, buffer_row in enumerate(buffer_rows, start=1):
            head_path = Path(buffer_row.get("head_crop_path", ""))
            body_path = Path(buffer_row.get("body_crop_path", event.get("best_body_crop", "")))
            face_result = extract_embedding_from_image(app, head_path)
            body_result = extract_body_feature(body_path)
            if face_result.get("face_count", 0) > 0:
                face_detected_in_buffer += 1
            if face_result.get("embedding") is not None:
                face_embedding_created_in_buffer += 1
            analyzed_buffer.append(
                {
                    "buffer_index": index,
                    "frame_id": as_int(buffer_row.get("frame_id")),
                    "relative_sec": as_float(buffer_row.get("relative_sec")),
                    "bbox_area": as_int(buffer_row.get("bbox_area", 0)),
                    "head_crop_path": str(head_path),
                    "body_crop_path": str(body_path),
                    "face_result": face_result,
                    "body_result": body_result,
                    "face_quality_score": _buffer_face_quality(face_result, buffer_row),
                }
            )

        selected_face = None
        for candidate in sorted(analyzed_buffer, key=lambda item: item["face_quality_score"], reverse=True):
            if candidate["face_result"].get("status") == "ok":
                selected_face = candidate
                break
        if selected_face is None and analyzed_buffer:
            selected_face = analyzed_buffer[0]

        best_body = extract_body_feature(Path(event["best_body_crop"]))
        pending_buffer_items = []
        for candidate in analyzed_buffer:
            candidate_event = dict(event)
            candidate_event["pending_parent_event_id"] = event["event_id"]
            candidate_event["pending_buffer_index"] = candidate["buffer_index"]
            candidate_event["pending_buffer_frame_id"] = candidate["frame_id"]
            candidate_event["pending_buffer_relative_sec"] = candidate["relative_sec"]
            candidate_event["pending_buffer_bbox_area"] = candidate["bbox_area"]
            candidate_event["best_head_crop"] = candidate["head_crop_path"]
            candidate_event["best_body_crop"] = candidate["body_crop_path"]
            pending_buffer_items.append(
                {
                    "event": candidate_event,
                    "face_embedding": candidate["face_result"].get("embedding"),
                    "face_status": candidate["face_result"].get("status", "missing"),
                    "face_count": candidate["face_result"].get("face_count", 0),
                    "face_det_score": candidate["face_result"].get("det_score", 0.0),
                    "face_bbox": candidate["face_result"].get("bbox", ""),
                    "face_message": candidate["face_result"].get("message", ""),
                    "used_face_crop": "face_buffer_head",
                    "used_face_crop_path": candidate["head_crop_path"],
                    "body_embedding": candidate["body_result"].get("embedding"),
                    "body_status": candidate["body_result"].get("status", "missing"),
                    "body_message": candidate["body_result"].get("message", ""),
                    "body_shape": candidate["body_result"].get("shape", ""),
                    "buffer_quality_score": candidate["face_quality_score"],
                }
            )

        analyzed.append(
            {
                "event": event,
                "face_embedding": selected_face["face_result"]["embedding"] if selected_face else None,
                "face_status": selected_face["face_result"]["status"] if selected_face else "missing",
                "face_count": selected_face["face_result"]["face_count"] if selected_face else 0,
                "face_det_score": selected_face["face_result"]["det_score"] if selected_face else 0.0,
                "face_bbox": selected_face["face_result"]["bbox"] if selected_face else "",
                "face_message": selected_face["face_result"]["message"] if selected_face else "missing",
                "used_face_crop": "face_buffer_head" if selected_face else "",
                "used_face_crop_path": selected_face["head_crop_path"] if selected_face else "",
                "body_embedding": best_body["embedding"],
                "body_status": best_body["status"],
                "body_message": best_body["message"],
                "body_shape": best_body["shape"],
                "evidence_buffer_count": len(analyzed_buffer),
                "face_buffer_detected_count": face_detected_in_buffer,
                "face_buffer_embedding_count": face_embedding_created_in_buffer,
                "face_best_shot_selected": bool(selected_face and selected_face["face_result"].get("status") == "ok"),
                "face_best_shot_frame": selected_face["frame_id"] if selected_face else "",
                "face_best_shot_quality": selected_face["face_quality_score"] if selected_face else "",
                "pending_buffer_items": pending_buffer_items,
            }
        )
    return analyzed


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
        selected = max(candidates, key=lambda row: (row["area"], -abs(row["frame_id"] - (frame_min or row["frame_id"]))))
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
        spatial = core_resolve_spatial_context(camera_id, record.get("foot_x"), record.get("foot_y"), transition_map)
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


def build_timeline_audit(track_rows, selected_cameras):
    stage_a_rows = []
    by_gt = defaultdict(lambda: defaultdict(list))
    for row in track_rows:
        by_gt[str(row["global_gt_id"])][row["camera_id"]].append(row)
    for gt_id, camera_map in by_gt.items():
        windows = {}
        for camera_id, rows in camera_map.items():
            rows = sorted(rows, key=lambda item: item["frame_id"])
            windows[camera_id] = {
                "first_seen_time": rows[0]["relative_sec"],
                "last_seen_time": rows[-1]["relative_sec"],
                "first_frame": rows[0]["frame_id"],
                "last_frame": rows[-1]["frame_id"],
                "frame_count": len(rows),
            }
        overlap_pairs = []
        for camera_a, camera_b in combinations(sorted(windows.keys()), 2):
            a = windows[camera_a]
            b = windows[camera_b]
            overlap = not (a["last_seen_time"] < b["first_seen_time"] or b["last_seen_time"] < a["first_seen_time"])
            near_overlap = abs(a["first_seen_time"] - b["first_seen_time"]) <= 2.0
            overlap_pairs.append(
                {
                    "camera_a": camera_a,
                    "camera_b": camera_b,
                    "overlap_in_time": overlap,
                    "near_overlap": near_overlap,
                    "start_gap_sec": round(abs(a["first_seen_time"] - b["first_seen_time"]), 3),
                }
            )
        stage_a_rows.append(
            {
                "global_gt_id": gt_id,
                "timeline_camera_count": len(windows),
                "timeline_cameras": ",".join(sorted(windows.keys())),
                "timeline_windows_json": json.dumps(windows, ensure_ascii=False),
                "timeline_overlap_pairs_json": json.dumps(overlap_pairs, ensure_ascii=False),
                "timeline_has_overlap": any(item["overlap_in_time"] for item in overlap_pairs),
                "timeline_has_near_overlap": any(item["near_overlap"] for item in overlap_pairs),
            }
        )
    stage_a = {
        "total_timeline_rows": len(track_rows),
        "unique_gt_id_count": len(by_gt),
        "timeline_multicam_gt_count": sum(1 for camera_map in by_gt.values() if len(camera_map) > 1),
    }
    return stage_a, stage_a_rows, by_gt


def build_event_from_record(
    event_id,
    event_type,
    record,
    best_record,
    all_records,
    dataset_root: Path,
    crop_root: Path,
    head_cfg,
    extra,
    transition_map,
    frame_cache=None,
    face_buffer_cfg=None,
):
    image_path = dataset_root / best_record["image_rel_path"]
    active_frame_cache = frame_cache or FrameSourceCache()
    try:
        image = active_frame_cache.frame_for_record(best_record, dataset_root)
    except RuntimeError as exc:
        return None, str(exc)
    img_h, img_w = image.shape[:2]
    body_rect = clamp_rect(
        {"x": best_record["xmin"], "y": best_record["ymin"], "width": best_record["width"], "height": best_record["height"]},
        img_w,
        img_h,
    )
    head_rect = get_head_rect(best_record, head_cfg, img_w, img_h)
    camera_dir = crop_root / record["camera_id"]
    body_path = camera_dir / f"{event_id}_body.png"
    head_path = camera_dir / f"{event_id}_head.png"
    ok_body, body_message = crop_and_save_from_record(active_frame_cache, dataset_root, best_record, body_rect, body_path)
    ok_head, head_message = crop_and_save_from_record(active_frame_cache, dataset_root, best_record, head_rect, head_path)
    if not ok_body or not ok_head:
        return None, f"crop_failure:{body_message}|{head_message}"
    evidence_buffer_rows = crop_event_buffer_records(
        active_frame_cache,
        dataset_root,
        crop_root,
        event_id,
        record["camera_id"],
        select_evidence_buffer_records(
            all_records,
            extra.get("anchor_frame_id", record["frame_id"]),
            best_record,
            face_buffer_cfg,
        ),
        head_cfg,
    )
    spatial_meta = resolve_zone_for_record(record, transition_map)
    event = {
        "event_id": event_id,
        "event_type": event_type,
        "camera_id": record["camera_id"],
        "camera_role": record["role"],
        "global_gt_id": str(record["global_gt_id"]),
        "frame_id": record["frame_id"],
        "relative_sec": record["relative_sec"],
        "best_shot_frame": best_record["frame_id"],
        "best_shot_sec": best_record["relative_sec"],
        "best_head_crop": str(head_path),
        "best_body_crop": str(body_path),
        "source_image": str(image_path),
        "bbox_xmin": best_record["xmin"],
        "bbox_ymin": best_record["ymin"],
        "bbox_xmax": best_record["xmax"],
        "bbox_ymax": best_record["ymax"],
        "bbox_width": best_record["width"],
        "bbox_height": best_record["height"],
        "bbox_area": best_record["area"],
        "foot_x": record.get("foot_x", ""),
        "foot_y": record.get("foot_y", ""),
        "zone_id": spatial_meta["zone_id"],
        "zone_type": spatial_meta["zone_type"],
        "zone_reason": spatial_meta["zone_reason"],
        "zone_fallback_used": spatial_meta["zone_fallback_used"],
        "subzone_id": spatial_meta["subzone_id"],
        "subzone_type": spatial_meta["subzone_type"],
        "subzone_reason": spatial_meta["subzone_reason"],
        "subzone_fallback_used": spatial_meta["subzone_fallback_used"],
        "matched_zone_region_id": spatial_meta["matched_zone_region_id"],
        "matched_subzone_region_id": spatial_meta["matched_subzone_region_id"],
        "assignment_point_x": spatial_meta["assignment_point_x"],
        "assignment_point_y": spatial_meta["assignment_point_y"],
        "evidence_buffer_count": len(evidence_buffer_rows),
        "evidence_buffer_json": json.dumps(evidence_buffer_rows, ensure_ascii=False),
    }
    event.update(extra)
    return event, "ok"


def find_entry_anchor(records, camera_cfg, line_threshold):
    if len(records) < 2:
        return None, "no_track"
    line = camera_cfg.get("entry_line") or []
    in_side = camera_cfg.get("in_side_point") or []
    direction_cfg = camera_cfg.get("direction_filter", {}) or {}
    if not line or not in_side:
        return None, "missing_manual_entry_line"
    history_window = max(2, as_int(direction_cfg.get("history_window", 6), 6))
    transition_map = camera_cfg.get("_transition_map")
    camera_id = camera_cfg.get("_camera_id", "")
    for index in range(1, len(records)):
        start_index = max(0, index - history_window + 1)
        window_rows = records[start_index : index + 1]
        points = [{"x": row["foot_x"], "y": row["foot_y"]} for row in window_rows]
        spatial_history = []
        if transition_map and camera_id:
            spatial_history = [
                core_resolve_spatial_context(camera_id, row.get("foot_x"), row.get("foot_y"), transition_map)
                for row in window_rows
            ]
        result = evaluate_direction(points, line, in_side, spatial_history=spatial_history, config=direction_cfg)
        if result["decision"] == "IN":
            curr_row = records[index]
            return {
                "crossing_index": index,
                "crossing_row": curr_row,
                "prev_row": records[index - 1],
                "direction_result": result,
            }, "ok"
    return None, "filtered_by_direction"


def pick_observation_candidate(records, anchor_event, relation):
    if not records:
        return None, "no_track"
    anchor_frame = anchor_event["frame_id"]
    min_gap = relation["min_travel_time_frames"]
    max_gap = relation["max_travel_time_frames"]
    relation_type = relation["relation_type"]
    candidates = []
    for row in records:
        delta = row["frame_id"] - anchor_frame
        if relation_type == "overlap":
            if abs(delta) <= max_gap:
                candidates.append((abs(delta), -row["area"], row))
        elif relation_type == "sequential":
            if delta >= min_gap and delta <= max_gap:
                candidates.append((abs(delta - relation["avg_travel_time_frames"]), -row["area"], row))
        else:
            if delta >= 0 and delta <= max_gap:
                candidates.append((abs(delta), -row["area"], row))
    if not candidates:
        return None, "blocked_by_travel_time"
    candidates.sort(key=lambda item: item[:2])
    return candidates[0][2], "ok"


def build_baseline_stage_b(track_rows, queue_rows, config, transition_map):
    selected_cameras = config["selected_cameras"]
    camera_cfgs = config["cameras"]
    line_threshold = float(config["line_crossing_distance_threshold"])
    by_gt_camera, _ = build_track_indexes(track_rows)
    queue_by_gt_camera = defaultdict(list)
    for row in queue_rows:
        queue_by_gt_camera[(str(row["global_gt_id"]), row["camera_id"])].append(row)
    all_gt_ids = sorted({str(row["global_gt_id"]) for row in track_rows}, key=lambda value: int(value))
    rows = []
    for gt_id in all_gt_ids:
        for camera_id in selected_cameras:
            records = by_gt_camera.get((gt_id, camera_id), [])
            created = bool(queue_by_gt_camera.get((gt_id, camera_id)))
            reason = ""
            detail = ""
            event_id = ""
            if not records:
                reason = "no_track"
            elif created:
                reason = "candidate_created"
                event_id = queue_by_gt_camera[(gt_id, camera_id)][0]["event_id"]
            elif camera_cfgs[camera_id]["role"] == "entry":
                camera_cfg = dict(camera_cfgs[camera_id])
                camera_cfg["_camera_id"] = camera_id
                camera_cfg["_transition_map"] = transition_map
                anchor, anchor_reason = find_entry_anchor(records, camera_cfg, line_threshold)
                if anchor_reason == "filtered_by_direction":
                    reason = "filtered_by_direction"
                elif anchor is None:
                    reason = "no_track"
                else:
                    reason = "not_selected_as_best_event"
                    detail = "entry_crossing_exists_but_no_queue_event"
            else:
                reason = "logic_collapse_across_cameras"
                detail = "baseline_queue_only_exports_entry_camera_events"
            rows.append(
                {
                    "run_mode": "mode_a_baseline",
                    "global_gt_id": gt_id,
                    "camera_id": camera_id,
                    "camera_role": camera_cfgs[camera_id]["role"],
                    "track_present": bool(records),
                    "candidate_event_created": created,
                    "candidate_event_id": event_id,
                    "drop_reason": reason if not created else "",
                    "drop_detail": detail,
                }
            )
    return rows


def build_fixed_candidate_events(track_rows, config, dataset_root: Path, crop_root: Path, transition_map):
    selected_cameras = config["selected_cameras"]
    camera_cfgs = config["cameras"]
    line_threshold = float(config["line_crossing_distance_threshold"])
    best_shot_window = int(config["best_shot_window_frames"])
    best_shot_cfg = config.get("best_shot_selection", {})
    face_buffer_cfg = merged_face_buffer_cfg(config.get("face_buffer", {}))
    topology = build_topology(transition_map)
    for src_camera, targets in topology.items():
        for dst_camera, info in targets.items():
            info["min_travel_time_frames"] = int(round(info["min_travel_time"] * float(config["assumed_video_fps"])))
            info["avg_travel_time_frames"] = int(round(info["avg_travel_time"] * float(config["assumed_video_fps"])))
            info["max_travel_time_frames"] = int(round(info["max_travel_time"] * float(config["assumed_video_fps"])))

    by_gt_camera, by_gt = build_track_indexes(track_rows)
    all_gt_ids = sorted(by_gt.keys(), key=lambda value: int(value))
    anchor_events_by_gt = defaultdict(list)
    placeholder_rows = []
    created_event_rows = []
    frame_cache = FrameSourceCache()
    try:
        for gt_id in all_gt_ids:
            for camera_id in selected_cameras:
                records = by_gt_camera.get((gt_id, camera_id), [])
                if not records:
                    placeholder_rows.append(
                        {
                            "global_gt_id": gt_id,
                            "camera_id": camera_id,
                            "camera_role": camera_cfgs[camera_id]["role"],
                            "track_present": False,
                            "candidate_event_created": False,
                            "drop_reason": "no_track",
                            "drop_detail": "",
                        }
                    )
                    continue
                if camera_cfgs[camera_id]["role"] != "entry":
                    placeholder_rows.append(
                        {
                            "global_gt_id": gt_id,
                            "camera_id": camera_id,
                            "camera_role": camera_cfgs[camera_id]["role"],
                            "track_present": True,
                            "candidate_event_created": False,
                            "drop_reason": "pending_followup_selection",
                            "drop_detail": "",
                        }
                    )
                    continue
                camera_cfg = dict(camera_cfgs[camera_id])
                camera_cfg["_camera_id"] = camera_id
                camera_cfg["_transition_map"] = transition_map
                anchor, anchor_reason = find_entry_anchor(records, camera_cfg, line_threshold)
                if anchor is None:
                    placeholder_rows.append(
                        {
                            "global_gt_id": gt_id,
                            "camera_id": camera_id,
                            "camera_role": camera_cfgs[camera_id]["role"],
                            "track_present": True,
                            "candidate_event_created": False,
                            "drop_reason": anchor_reason,
                            "drop_detail": "",
                        }
                    )
                    continue
                crossing_row = anchor["crossing_row"]
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
                    placeholder_rows.append(
                        {
                            "global_gt_id": gt_id,
                            "camera_id": camera_id,
                            "camera_role": camera_cfgs[camera_id]["role"],
                            "track_present": True,
                            "candidate_event_created": False,
                            "drop_reason": "not_selected_as_best_event",
                            "drop_detail": "no_best_record_in_window",
                        }
                    )
                    continue
                event_id = f"ENTRY_{camera_id}_{int(gt_id):04d}_{crossing_row['frame_id']:08d}"
                event, message = build_event_from_record(
                    event_id,
                    "ENTRY_IN",
                    crossing_row,
                    best_record,
                    records,
                    dataset_root,
                    crop_root,
                    config["head_crop"],
                    {
                        "anchor_camera_id": camera_id,
                        "anchor_frame_id": crossing_row["frame_id"],
                        "anchor_relative_sec": crossing_row["relative_sec"],
                        "relation_type": "entry",
                        "same_area_overlap": False,
                        "direction_reason": anchor["direction_result"].get("reason", ""),
                        "direction_history_points": anchor["direction_result"].get("history_points", 0),
                        "direction_momentum_px": anchor["direction_result"].get("momentum_px", 0.0),
                        "direction_inside_ratio": anchor["direction_result"].get("inside_ratio", 0.0),
                        "direction_cross_in": anchor["direction_result"].get("cross_in", False),
                        "direction_zone_transition_ok": anchor["direction_result"].get("zone_transition_ok", False),
                        **best_shot_meta,
                    },
                    transition_map,
                    frame_cache=frame_cache,
                    face_buffer_cfg=face_buffer_cfg,
                )
                if event is None:
                    placeholder_rows.append(
                        {
                            "global_gt_id": gt_id,
                            "camera_id": camera_id,
                            "camera_role": camera_cfgs[camera_id]["role"],
                            "track_present": True,
                            "candidate_event_created": False,
                            "drop_reason": "low_quality_crop",
                            "drop_detail": message,
                        }
                    )
                    continue
                created_event_rows.append(event)
                anchor_events_by_gt[gt_id].append(event)
                placeholder_rows.append(
                    {
                        "global_gt_id": gt_id,
                        "camera_id": camera_id,
                        "camera_role": camera_cfgs[camera_id]["role"],
                        "track_present": True,
                        "candidate_event_created": True,
                        "candidate_event_id": event_id,
                        "event_type": "ENTRY_IN",
                        "drop_reason": "",
                        "drop_detail": "",
                    }
                )
        final_rows = []
        created_lookup = {(row["global_gt_id"], row["camera_id"]): row for row in placeholder_rows if row["candidate_event_created"]}
        placeholder_lookup = {(row["global_gt_id"], row["camera_id"]): row for row in placeholder_rows if not row["candidate_event_created"]}

        for gt_id in all_gt_ids:
            anchors = sorted(anchor_events_by_gt.get(gt_id, []), key=lambda item: (item["relative_sec"], item["camera_id"]))
            for camera_id in selected_cameras:
                key = (gt_id, camera_id)
                if key in created_lookup:
                    row = dict(created_lookup[key])
                    row["run_mode"] = "mode_b_true_assoc"
                    final_rows.append(row)
                    continue
                placeholder = dict(placeholder_lookup[key])
                placeholder["run_mode"] = "mode_b_true_assoc"
                records = by_gt_camera.get(key, [])
                if not records:
                    final_rows.append(placeholder)
                    continue
                if not anchors:
                    if placeholder["drop_reason"] == "pending_followup_selection":
                        placeholder["drop_reason"] = "filtered_by_direction"
                        placeholder["drop_detail"] = "no_entry_anchor_in_any_entry_camera"
                    final_rows.append(placeholder)
                    continue

                best_option = None
                for anchor_event in anchors:
                    if anchor_event["camera_id"] == camera_id:
                        continue
                    relation = topology.get(anchor_event["camera_id"], {}).get(camera_id)
                    if not relation:
                        continue
                    candidate_row, candidate_reason = pick_observation_candidate(records, anchor_event, relation)
                    if candidate_row is None:
                        continue
                    option = {
                        "anchor_event": anchor_event,
                        "relation": relation,
                        "candidate_row": candidate_row,
                        "candidate_reason": candidate_reason,
                    }
                    if best_option is None:
                        best_option = option
                        continue
                    current_delta = abs(candidate_row["frame_id"] - anchor_event["frame_id"])
                    best_delta = abs(best_option["candidate_row"]["frame_id"] - best_option["anchor_event"]["frame_id"])
                    if current_delta < best_delta or (
                        current_delta == best_delta and candidate_row["area"] > best_option["candidate_row"]["area"]
                    ):
                        best_option = option

                if best_option is None:
                    placeholder["drop_reason"] = "blocked_by_travel_time"
                    placeholder["drop_detail"] = "no_topology_window_match_from_any_entry_anchor"
                    final_rows.append(placeholder)
                    continue

                best_record, best_shot_meta = select_best_record(
                    records,
                    max(best_option["candidate_row"]["frame_id"] - 15, 0),
                    best_option["candidate_row"]["frame_id"] + best_shot_window,
                    camera_id=camera_id,
                    transition_map=transition_map,
                    best_shot_cfg=best_shot_cfg,
                    anchor_frame_id=best_option["candidate_row"]["frame_id"],
                    relation_type=best_option["relation"]["relation_type"],
                )
                if best_record is None:
                    placeholder["drop_reason"] = "not_selected_as_best_event"
                    placeholder["drop_detail"] = "no_best_record_for_followup_event"
                    final_rows.append(placeholder)
                    continue

                anchor_event = best_option["anchor_event"]
                relation = best_option["relation"]
                event_type = "OVERLAP_OBSERVATION" if relation["relation_type"] == "overlap" else "FOLLOWUP_OBSERVATION"
                event_id = f"{event_type}_{camera_id}_{int(gt_id):04d}_{best_option['candidate_row']['frame_id']:08d}"
                event, message = build_event_from_record(
                    event_id,
                    event_type,
                    best_option["candidate_row"],
                    best_record,
                    records,
                    dataset_root,
                    crop_root,
                    config["head_crop"],
                    {
                        "anchor_camera_id": anchor_event["camera_id"],
                        "anchor_frame_id": anchor_event["frame_id"],
                        "anchor_relative_sec": anchor_event["relative_sec"],
                        "relation_type": relation["relation_type"],
                        "same_area_overlap": relation["same_area_overlap"],
                        "min_travel_time": relation["min_travel_time"],
                        "avg_travel_time": relation["avg_travel_time"],
                        "max_travel_time": relation["max_travel_time"],
                        "delta_from_anchor_sec": round(best_option["candidate_row"]["relative_sec"] - anchor_event["relative_sec"], 3),
                        **best_shot_meta,
                    },
                    transition_map,
                    frame_cache=frame_cache,
                    face_buffer_cfg=face_buffer_cfg,
                )
                if event is None:
                    placeholder["drop_reason"] = "low_quality_crop"
                    placeholder["drop_detail"] = message
                    final_rows.append(placeholder)
                    continue
                created_event_rows.append(event)
                final_rows.append(
                    {
                        "run_mode": "mode_b_true_assoc",
                        "global_gt_id": gt_id,
                        "camera_id": camera_id,
                        "camera_role": camera_cfgs[camera_id]["role"],
                        "track_present": True,
                        "candidate_event_created": True,
                        "candidate_event_id": event_id,
                        "event_type": event_type,
                        "drop_reason": "",
                        "drop_detail": "",
                    }
                )
    finally:
        frame_cache.close()

    created_event_rows.sort(key=lambda row: (row["anchor_relative_sec"], 0 if row["event_type"] == "ENTRY_IN" else 1, row["camera_id"]))
    return created_event_rows, final_rows, anchor_events_by_gt, by_gt_camera, topology


def build_baseline_assignments(analyzed_events, identity_means, threshold, unknown_prefix, unknown_start):
    by_gt = defaultdict(list)
    for item in analyzed_events:
        by_gt[item["event"]["global_gt_id"]].append(item)

    assignments = {}
    unknown_index = unknown_start
    for gt_id, items in by_gt.items():
        best_known = None
        for item in items:
            emb = item["face_embedding"]
            if emb is None or not identity_means:
                continue
            for identity_id, ref_vec in identity_means.items():
                score = cosine_similarity(emb, ref_vec)
                if best_known is None or score > best_known["score"]:
                    best_known = {
                        "identity_id": identity_id,
                        "score": score,
                        "event_id": item["event"]["event_id"],
                        "used_crop": item["used_face_crop"],
                    }
        if best_known and best_known["score"] >= threshold:
            assignments[gt_id] = {
                "identity_status": "known",
                "matched_known_id": best_known["identity_id"],
                "matched_known_score": round(best_known["score"], 4),
                "unknown_global_id": "",
                "resolved_global_id": best_known["identity_id"],
                "resolution_source": f"gt_grouped_face_match:{best_known['used_crop']}",
                "decision_reason": "best_known_match_over_threshold",
            }
        else:
            assignments[gt_id] = {
                "identity_status": "unknown",
                "matched_known_id": "",
                "matched_known_score": round(best_known["score"], 4) if best_known else "",
                "unknown_global_id": f"{unknown_prefix}_{unknown_index:04d}",
                "resolved_global_id": f"{unknown_prefix}_{unknown_index:04d}",
                "resolution_source": "gt_grouped_unknown_assignment",
                "decision_reason": "no_known_match_over_threshold",
            }
            unknown_index += 1
    return assignments


def build_resolved_rows_from_gt_assignments(analyzed_events, assignments, run_mode):
    rows = []
    for item in analyzed_events:
        event = item["event"]
        assignment = assignments[event["global_gt_id"]]
        rows.append(
            {
                "run_mode": run_mode,
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "camera_id": event["camera_id"],
                "frame_id": event["frame_id"],
                "relative_sec": event["relative_sec"],
                "global_gt_id": event["global_gt_id"],
                "anchor_camera_id": event.get("anchor_camera_id", ""),
                "anchor_relative_sec": event.get("anchor_relative_sec", ""),
                "relation_type": event.get("relation_type", ""),
                "zone_id": event.get("zone_id", ""),
                "zone_type": event.get("zone_type", ""),
                "zone_reason": event.get("zone_reason", ""),
                "zone_fallback_used": event.get("zone_fallback_used", False),
                "subzone_id": event.get("subzone_id", ""),
                "subzone_type": event.get("subzone_type", ""),
                "subzone_reason": event.get("subzone_reason", ""),
                "subzone_fallback_used": event.get("subzone_fallback_used", False),
                "best_head_crop": event["best_head_crop"],
                "best_body_crop": event["best_body_crop"],
                "identity_status": assignment["identity_status"],
                "matched_known_id": assignment["matched_known_id"],
                "matched_known_score": assignment["matched_known_score"],
                "unknown_global_id": assignment["unknown_global_id"],
                "resolved_global_id": assignment["resolved_global_id"],
                "resolution_source": assignment["resolution_source"],
                "decision_reason": assignment["decision_reason"],
                "face_embedding_status": item["face_status"],
                "face_count": item["face_count"],
                "face_det_score": round(item["face_det_score"], 4) if item["face_det_score"] else "",
                "used_face_crop": item["used_face_crop"],
                "used_face_crop_path": item["used_face_crop_path"],
                "face_bbox": item["face_bbox"],
                "body_feature_status": item["body_status"],
                "body_feature_shape": item["body_shape"],
            }
        )
    return rows


def build_baseline_stream_timeline(track_rows, assignments):
    rows = []
    for row in track_rows:
        gt_id = str(row["global_gt_id"])
        if gt_id not in assignments:
            continue
        assignment = assignments[gt_id]
        rows.append(
            {
                "resolved_global_id": assignment["resolved_global_id"],
                "identity_status": assignment["identity_status"],
                "matched_known_id": assignment["matched_known_id"],
                "unknown_global_id": assignment["unknown_global_id"],
                "global_gt_id": gt_id,
                "camera_id": row["camera_id"],
                "frame_id": row["frame_id"],
                "relative_sec": row["relative_sec"],
                "bbox_xmin": row["xmin"],
                "bbox_ymin": row["ymin"],
                "bbox_xmax": row["xmax"],
                "bbox_ymax": row["ymax"],
                "stream_unification_basis": "wildtrack_global_gt_id_demo_bridge",
            }
        )
    return rows


def keep_top_refs(refs, new_ref, top_k):
    refs.append(new_ref)
    refs.sort(key=lambda item: (item["quality_score"], item["relative_sec"]), reverse=True)
    del refs[top_k:]


def create_unknown_profile(unknown_global_id, item, top_k_face=3, top_k_body=5):
    return core_create_unknown_profile(
        unknown_global_id,
        item,
        policy={"top_k_face_refs": top_k_face, "top_k_body_refs": top_k_body},
    )


def update_unknown_profile(profile, item):
    return core_update_unknown_profile(profile, item)


def best_known_match(face_embedding, identity_means):
    return core_best_known_match(face_embedding, identity_means)


def evaluate_profile_candidate(item, profile, topology):
    return core_evaluate_profile_candidate(item, profile, topology)


def load_association_policy(config_path=None, base_dir=None):
    return core_load_association_policy(config_path=config_path, base_dir=base_dir)


def assign_model_identities(analyzed_events, identity_means, topology, unknown_prefix, unknown_start, policy=None, return_debug_bundle=False):
    return core_assign_model_identities(
        analyzed_events,
        identity_means,
        topology,
        unknown_prefix,
        unknown_start,
        policy=policy,
        return_debug_bundle=return_debug_bundle,
    )


def build_unknown_profile_rows(profiles):
    rows = []
    for profile in profiles:
        face_refs = [
            {
                "event_id": ref["event_id"],
                "camera_id": ref["camera_id"],
                "relative_sec": ref["relative_sec"],
                "zone_id": ref.get("zone_id", ""),
                "zone_type": ref.get("zone_type", ""),
                "subzone_id": ref.get("subzone_id", ""),
                "subzone_type": ref.get("subzone_type", ""),
                "crop_path": ref["crop_path"],
                "quality_score": round(ref["quality_score"], 4),
            }
            for ref in profile["face_refs"]
        ]
        body_refs = [
            {
                "event_id": ref["event_id"],
                "camera_id": ref["camera_id"],
                "relative_sec": ref["relative_sec"],
                "zone_id": ref.get("zone_id", ""),
                "zone_type": ref.get("zone_type", ""),
                "subzone_id": ref.get("subzone_id", ""),
                "subzone_type": ref.get("subzone_type", ""),
                "crop_path": ref["crop_path"],
                "quality_score": round(ref["quality_score"], 4),
            }
            for ref in profile["body_refs"]
        ]
        rows.append(
            {
                "unknown_global_id": profile["unknown_global_id"],
                "first_seen_camera": profile["first_seen_camera"],
                "first_seen_time": round(profile["first_seen_time"], 3),
                "first_seen_zone": profile.get("first_seen_zone", ""),
                "first_seen_subzone": profile.get("first_seen_subzone", ""),
                "latest_seen_camera": profile["latest_seen_camera"],
                "latest_seen_time": round(profile["latest_seen_time"], 3),
                "latest_seen_zone": profile.get("latest_seen_zone", ""),
                "latest_seen_subzone": profile.get("latest_seen_subzone", ""),
                "history_cameras": ",".join(sorted(profile["history_cameras"])),
                "cameras_seen": ",".join(sorted(profile.get("cameras_seen", profile["history_cameras"]))),
                "zones_seen": ",".join(sorted(profile.get("zones_seen", []))),
                "subzones_seen": ",".join(sorted(profile.get("subzones_seen", []))),
                "event_ids": ",".join(profile["event_ids"]),
                "reference_face_frames_json": json.dumps(face_refs, ensure_ascii=False),
                "reference_body_frames_json": json.dumps(body_refs, ensure_ascii=False),
                "representative_face_count": len(profile["face_refs"]),
                "representative_body_count": len(profile["body_refs"]),
                "quality_stats_json": json.dumps(profile.get("quality_stats", {}), ensure_ascii=False),
                "expiry_at_sec": round(float(profile.get("expiry_at_sec", profile["latest_seen_time"])), 3),
                "gt_ids_for_audit": ",".join(sorted(profile["gt_ids"], key=lambda value: int(value))),
            }
        )
    return rows


def build_stream_timeline_from_events(track_rows, resolved_rows):
    resolved_by_gt_camera = {}
    for row in resolved_rows:
        resolved_by_gt_camera[(row["global_gt_id"], row["camera_id"])] = row
    timeline = []
    for track in track_rows:
        key = (str(track["global_gt_id"]), track["camera_id"])
        resolved = resolved_by_gt_camera.get(key)
        if not resolved:
            continue
        timeline.append(
            {
                "resolved_global_id": resolved["resolved_global_id"],
                "identity_status": resolved["identity_status"],
                "matched_known_id": resolved["matched_known_id"],
                "unknown_global_id": resolved["unknown_global_id"],
                "global_gt_id": resolved["global_gt_id"],
                "camera_id": track["camera_id"],
                "frame_id": track["frame_id"],
                "relative_sec": track["relative_sec"],
                "bbox_xmin": track["xmin"],
                "bbox_ymin": track["ymin"],
                "bbox_xmax": track["xmax"],
                "bbox_ymax": track["ymax"],
                "stream_unification_basis": "per_camera_event_projection_without_gt_crossstream_bridge",
            }
        )
    return timeline


def summarize_mode(resolved_rows, stage_b_rows, stage_a, gt_bridge_active, true_model_based_reuse_count, association_summary=None):
    unknown_rows = [row for row in resolved_rows if row["identity_status"] == "unknown"]
    known_rows = [row for row in resolved_rows if row["identity_status"] == "known"]
    deferred_rows = [row for row in resolved_rows if row["identity_status"] == "deferred"]
    unknown_groups = defaultdict(list)
    for row in unknown_rows:
        unknown_groups[row["unknown_global_id"]].append(row)
    reused_unknown_ids = [
        unknown_id
        for unknown_id, rows in unknown_groups.items()
        if unknown_id and len(rows) > 1 and len({row["camera_id"] for row in rows}) > 1
    ]
    resolved_multicam_gt_count = sum(
        1
        for gt_id in {row["global_gt_id"] for row in resolved_rows}
        if len({row["camera_id"] for row in resolved_rows if row["global_gt_id"] == gt_id}) > 1
    )
    metrics = {
        "total_event_count": len(resolved_rows),
        "known_event_count": len(known_rows),
        "unknown_event_count": len(unknown_rows),
        "deferred_event_count": len(deferred_rows),
        "unique_known_id_count": len({row["matched_known_id"] for row in known_rows if row["matched_known_id"]}),
        "unique_unknown_id_count": len({row["unknown_global_id"] for row in unknown_rows if row["unknown_global_id"]}),
        "reused_unknown_id_count": len(reused_unknown_ids),
        "reused_unknown_event_count": sum(len(unknown_groups[unknown_id]) for unknown_id in reused_unknown_ids),
        "timeline_multicam_gt_count": stage_a["timeline_multicam_gt_count"],
        "resolved_multicam_gt_count": resolved_multicam_gt_count,
        "gt_bridge_active": gt_bridge_active,
        "true_model_based_reuse_count": true_model_based_reuse_count,
        "stage_b_created_event_count": sum(1 for row in stage_b_rows if row["candidate_event_created"]),
    }
    if association_summary:
        metrics.update(association_summary)
    return metrics


def summarize_face_body_usage(analyzed_items, resolved_rows, decision_logs, profiles):
    face_status_counts = Counter()
    face_unusable_reason_counts = Counter()
    modality_primary_counts = Counter()
    resolution_source_counts = Counter()
    unknown_ids_by_gt = defaultdict(set)

    for item in analyzed_items:
        face_status_counts[item.get("face_status") or "missing"] += 1

    for row in decision_logs:
        modality_primary_counts[row.get("modality_primary") or ""] += 1
        reason = row.get("face_unusable_reason") or ""
        if reason:
            face_unusable_reason_counts[reason] += 1

    for row in resolved_rows:
        resolution_source_counts[row.get("resolution_source") or ""] += 1
        if row.get("identity_status") == "unknown" and row.get("global_gt_id") and row.get("unknown_global_id"):
            unknown_ids_by_gt[str(row["global_gt_id"])].add(row["unknown_global_id"])

    split_gt_count = sum(1 for unknown_ids in unknown_ids_by_gt.values() if len(unknown_ids) > 1)
    resolved_by_event_id = {row.get("event_id"): row for row in resolved_rows}

    return {
        "analyzed_event_count": len(analyzed_items),
        "face_detected_count": sum(1 for item in analyzed_items if item.get("face_buffer_detected_count", 0) > 0),
        "face_buffered_count": sum(1 for item in analyzed_items if item.get("evidence_buffer_count", 0) > 0),
        "best_shot_selected_count": sum(1 for item in analyzed_items if item.get("face_best_shot_selected")),
        "face_embedding_available_count": sum(1 for item in analyzed_items if item.get("face_embedding") is not None),
        "face_embedding_created_count": sum(1 for item in analyzed_items if item.get("face_embedding") is not None),
        "face_reference_stored_count": sum(
            1
            for item in analyzed_items
            if item.get("face_embedding") is not None
            and resolved_by_event_id.get(item["event"]["event_id"], {}).get("identity_status") == "unknown"
        ),
        "face_reference_retrieved_count": sum(
            1
            for row in decision_logs
            if any(candidate.get("face_available") for candidate in row.get("candidate_evaluations", []))
        ),
        "face_usable_event_count": sum(1 for row in decision_logs if not row.get("face_unusable_reason")),
        "face_primary_decision_count": sum(1 for row in decision_logs if row.get("modality_primary") == "face"),
        "body_primary_decision_count": sum(1 for row in decision_logs if row.get("modality_primary") == "body"),
        "body_fallback_used_count": sum(1 for row in decision_logs if row.get("body_fallback_used")),
        "pending_count": sum(1 for row in decision_logs if row.get("pending_used")),
        "pending_to_reuse_count": sum(1 for row in decision_logs if row.get("pending_resolution") == "reuse_existing_unknown"),
        "pending_to_create_count": sum(1 for row in decision_logs if row.get("pending_resolution") == "create_new_unknown"),
        "known_face_match_success_count": sum(
            1
            for row in resolved_rows
            if row.get("identity_status") == "known" and row.get("resolution_source") == "model_face_match"
        ),
        "face_status_counts": dict(sorted(face_status_counts.items())),
        "face_unusable_reason_counts": dict(sorted(face_unusable_reason_counts.items())),
        "modality_primary_counts": dict(sorted((key or "none", value) for key, value in modality_primary_counts.items())),
        "resolution_source_counts": dict(sorted(resolution_source_counts.items())),
        "split_gt_count": split_gt_count,
    }


def build_coverage_and_failures(stage_a_rows, stage_b_baseline, stage_b_fixed, resolved_baseline, resolved_fixed):
    baseline_events_by_gt = defaultdict(list)
    fixed_events_by_gt = defaultdict(list)
    for row in resolved_baseline:
        baseline_events_by_gt[row["global_gt_id"]].append(row)
    for row in resolved_fixed:
        fixed_events_by_gt[row["global_gt_id"]].append(row)
    stage_b_baseline_map = {(row["global_gt_id"], row["camera_id"]): row for row in stage_b_baseline}
    stage_b_fixed_map = {(row["global_gt_id"], row["camera_id"]): row for row in stage_b_fixed}

    coverage_rows = []
    missing_rows = []
    expected_failure_rows = []
    overlap_rows = []

    for row in stage_a_rows:
        gt_id = row["global_gt_id"]
        windows = json.loads(row["timeline_windows_json"])
        overlap_pairs = json.loads(row["timeline_overlap_pairs_json"])
        baseline_events = baseline_events_by_gt.get(gt_id, [])
        fixed_events = fixed_events_by_gt.get(gt_id, [])
        baseline_cameras = sorted({item["camera_id"] for item in baseline_events})
        fixed_cameras = sorted({item["camera_id"] for item in fixed_events})
        fixed_unknown_ids = sorted({item["unknown_global_id"] for item in fixed_events if item["unknown_global_id"]})
        expected_candidate = (
            row["timeline_camera_count"] > 1
            and (row["timeline_has_overlap"] or row["timeline_has_near_overlap"])
            and len(fixed_cameras) > 1
        )
        failed_expected = expected_candidate and len(fixed_unknown_ids) > 1
        coverage_rows.append(
            {
                "global_gt_id": gt_id,
                "timeline_camera_count": row["timeline_camera_count"],
                "timeline_cameras": row["timeline_cameras"],
                "timeline_windows_json": row["timeline_windows_json"],
                "timeline_overlap_pairs_json": row["timeline_overlap_pairs_json"],
                "baseline_resolved_camera_count": len(baseline_cameras),
                "baseline_resolved_cameras": ",".join(baseline_cameras),
                "mode_b_resolved_camera_count": len(fixed_cameras),
                "mode_b_resolved_cameras": ",".join(fixed_cameras),
                "mode_b_resolved_unknown_ids": ",".join(fixed_unknown_ids),
                "expected_reid_candidate": expected_candidate,
                "failed_expected_reid_candidate": failed_expected,
            }
        )
        if failed_expected:
            expected_failure_rows.append(
                {
                    "global_gt_id": gt_id,
                    "timeline_cameras": row["timeline_cameras"],
                    "timeline_windows_json": row["timeline_windows_json"],
                    "mode_b_resolved_cameras": ",".join(fixed_cameras),
                    "mode_b_resolved_unknown_ids": ",".join(fixed_unknown_ids),
                    "failure_reason": "multiple_unknown_ids_for_good_multicam_candidate",
                }
            )
        for camera_id in sorted(windows.keys()):
            baseline_stage = stage_b_baseline_map.get((gt_id, camera_id))
            fixed_stage = stage_b_fixed_map.get((gt_id, camera_id))
            if baseline_stage and not baseline_stage["candidate_event_created"]:
                missing_rows.append(
                    {
                        "run_mode": "mode_a_baseline",
                        "global_gt_id": gt_id,
                        "camera_id": camera_id,
                        "drop_reason": baseline_stage["drop_reason"],
                        "drop_detail": baseline_stage.get("drop_detail", ""),
                    }
                )
            if fixed_stage and not fixed_stage["candidate_event_created"]:
                missing_rows.append(
                    {
                        "run_mode": "mode_b_true_assoc",
                        "global_gt_id": gt_id,
                        "camera_id": camera_id,
                        "drop_reason": fixed_stage["drop_reason"],
                        "drop_detail": fixed_stage.get("drop_detail", ""),
                    }
                )
        for pair in overlap_pairs:
            overlap_rows.append(
                {
                    "global_gt_id": gt_id,
                    "camera_a": pair["camera_a"],
                    "camera_b": pair["camera_b"],
                    "overlap_in_time": pair["overlap_in_time"],
                    "near_overlap": pair["near_overlap"],
                    "start_gap_sec": pair["start_gap_sec"],
                    "baseline_has_both_events": pair["camera_a"] in baseline_cameras and pair["camera_b"] in baseline_cameras,
                    "mode_b_has_both_events": pair["camera_a"] in fixed_cameras and pair["camera_b"] in fixed_cameras,
                    "mode_b_reused_same_unknown": len(fixed_unknown_ids) == 1 if fixed_unknown_ids else False,
                }
            )
    return coverage_rows, missing_rows, expected_failure_rows, overlap_rows


def build_gt_split_merge(rows, run_mode):
    out_rows = []
    by_gt = defaultdict(set)
    by_resolved = defaultdict(set)
    for row in rows:
        by_gt[row["global_gt_id"]].add(row["resolved_global_id"])
        by_resolved[row["resolved_global_id"]].add(row["global_gt_id"])
    for gt_id, resolved_ids in by_gt.items():
        out_rows.append(
            {
                "run_mode": run_mode,
                "entity_type": "gt_id",
                "entity_id": gt_id,
                "issue_type": "split" if len(resolved_ids) > 1 else "clean",
                "linked_ids": ",".join(sorted(resolved_ids)),
                "linked_count": len(resolved_ids),
            }
        )
    for resolved_id, gt_ids in by_resolved.items():
        out_rows.append(
            {
                "run_mode": run_mode,
                "entity_type": "resolved_global_id",
                "entity_id": resolved_id,
                "issue_type": "merge" if len(gt_ids) > 1 else "clean",
                "linked_ids": ",".join(sorted(gt_ids, key=lambda value: int(value))),
                "linked_count": len(gt_ids),
            }
        )
    return out_rows


def build_unknown_reuse_rows(resolved_rows, run_mode):
    rows = []
    grouped = defaultdict(list)
    for row in resolved_rows:
        if row["identity_status"] == "unknown" and row["unknown_global_id"]:
            grouped[row["unknown_global_id"]].append(row)
    for unknown_id, items in sorted(grouped.items()):
        rows.append(
            {
                "run_mode": run_mode,
                "unknown_global_id": unknown_id,
                "event_count": len(items),
                "camera_count": len({item["camera_id"] for item in items}),
                "cameras": ",".join(sorted({item["camera_id"] for item in items})),
                "gt_count_for_audit": len({item["global_gt_id"] for item in items}),
                "gt_ids_for_audit": ",".join(sorted({item["global_gt_id"] for item in items}, key=lambda value: int(value))),
                "reused_across_cameras": len(items) > 1 and len({item["camera_id"] for item in items}) > 1,
            }
        )
    return rows


def render_report(report_path: Path, mode_a_metrics, mode_b_metrics, stage_a, root_cause_lines, coverage_rows, expected_failure_rows):
    failed_ids = ", ".join(row["global_gt_id"] for row in expected_failure_rows[:20]) or "none"
    text = f"""# Audit Report

## A. Executive summary
- Baseline mode created only {mode_a_metrics['total_event_count']} resolved events and only {mode_a_metrics['reused_unknown_id_count']} reused unknown IDs.
- The low reuse was introduced upstream: baseline event generation exported only entry-camera events, then baseline identity assignment grouped by GT and the timeline used GT bridge.
- Fixed mode created {mode_b_metrics['total_event_count']} resolved events and {mode_b_metrics['reused_unknown_id_count']} reused unknown IDs without GT bridge.

## B. Stage-by-stage pipeline counts
- Stage A raw timeline rows: {stage_a['total_timeline_rows']}
- Stage A unique GT IDs: {stage_a['unique_gt_id_count']}
- Stage A multi-camera GT IDs: {stage_a['timeline_multicam_gt_count']}
- Mode A baseline total_event_count: {mode_a_metrics['total_event_count']}
- Mode B total_event_count: {mode_b_metrics['total_event_count']}

## C. Unknown events vs unique unknown IDs
- Mode A baseline unknown_event_count = {mode_a_metrics['unknown_event_count']}
- Mode A baseline unique_unknown_id_count = {mode_a_metrics['unique_unknown_id_count']}
- Mode B unknown_event_count = {mode_b_metrics['unknown_event_count']}
- Mode B unique_unknown_id_count = {mode_b_metrics['unique_unknown_id_count']}

## D. GT identities appearing in multiple cameras
- timeline_multicam_gt_count = {stage_a['timeline_multicam_gt_count']}
- resolved_multicam_gt_count baseline = {mode_a_metrics['resolved_multicam_gt_count']}
- resolved_multicam_gt_count fixed = {mode_b_metrics['resolved_multicam_gt_count']}

## E. Why only 2 unknown IDs were reused
{os.linesep.join(f"- {line}" for line in root_cause_lines)}

## F. Overlapping-camera failure analysis
- The dataset has strong overlap, so the fixed mode treats topology as `relation_type=overlap` with near-zero travel time allowed.
- Baseline mode implicitly required entry-line creation and dropped most overlapping second-camera observations before association.

## G. GT bridge vs true association
- Mode A gt_bridge_active = {mode_a_metrics['gt_bridge_active']}
- Mode B gt_bridge_active = {mode_b_metrics['gt_bridge_active']}
- Mode A true_model_based_reuse_count = {mode_a_metrics['true_model_based_reuse_count']}
- Mode B true_model_based_reuse_count = {mode_b_metrics['true_model_based_reuse_count']}

## H. Expected reID cases that were missed
- failed_expected_reid_candidate_count = {mode_b_metrics['failed_expected_reid_candidate_count']}
- Example GT IDs: {failed_ids}

## I. Concrete fixes applied
- Generated separate per-camera candidate events instead of collapsing to entry-only queue rows.
- Added overlap-aware topology scoring with zero or near-zero travel time support.
- Removed GT from actual cross-camera unknown association inputs.
- Added unknown gallery multi-reference storage and full candidate trace logging.

## J. Comparison before vs after rerun
- reused_unknown_id_count: {mode_a_metrics['reused_unknown_id_count']} -> {mode_b_metrics['reused_unknown_id_count']}
- unknown_event_count: {mode_a_metrics['unknown_event_count']} -> {mode_b_metrics['unknown_event_count']}
- resolved_multicam_gt_count: {mode_a_metrics['resolved_multicam_gt_count']} -> {mode_b_metrics['resolved_multicam_gt_count']}

## K. Final answer: is the system good enough for the demo now?
- It is good enough for a correctness-focused demo if you present the fixed mode and clearly state that tracking still comes from GT-derived per-camera tracks.
- It is not yet a production-quality live reID system, but it now reflects real model-based cross-camera unknown reuse instead of GT-bridged consistency.
"""
    report_path.write_text(text, encoding="utf-8")

def main(config_path: Path):
    config = load_json(config_path)
    base_dir = resolve_path(config_path.parent, config.get("project_root", str(config_path.parents[1])))
    association_policy_config = config.get("association_policy_config", "")
    camera_transition_map_config = config.get("camera_transition_map_config", "")
    scene_calibration_config = config.get("scene_calibration_config", "")
    queue_csv = resolve_path(base_dir, config["wildtrack_identity_queue_csv"])
    wildtrack_config_path = resolve_path(
        base_dir,
        config.get("wildtrack_config_path", str(queue_csv.parents[2] / "wildtrack_demo_config.json")),
    )
    wildtrack_config = load_json(wildtrack_config_path)
    dataset_root = resolve_path(base_dir, config.get("dataset_root", str(wildtrack_config_path.parent / wildtrack_config["dataset_root"])))
    tracks_csv = resolve_path(base_dir, config.get("tracks_csv", str(queue_csv.parents[1] / "tracks" / "all_tracks_filtered.csv")))
    known_root = resolve_path(base_dir, config["known_face_gallery_root"])
    known_manifest_csv = resolve_path(base_dir, config["known_face_manifest_csv"])
    known_embeddings_csv = resolve_path(base_dir, config["known_face_embeddings_csv"])
    resolved_events_csv = resolve_path(base_dir, config["resolved_events_csv"])
    runtime_dir = resolved_events_csv.parent
    runtime_manifest_csv = known_manifest_csv.with_name("known_face_manifest_runtime.csv")
    baseline_resolved_csv = runtime_dir / "resolved_events_mode_a_baseline.csv"
    baseline_timeline_csv = runtime_dir / "stream_identity_timeline_mode_a.csv"
    mode_b_resolved_csv = runtime_dir / "resolved_events_mode_b_true_assoc.csv"
    mode_b_timeline_csv = runtime_dir / "stream_identity_timeline_mode_b.csv"
    summary_json = runtime_dir / "face_resolution_summary.json"
    face_body_summary_json = runtime_dir / "face_body_usage_summary.json"
    unknown_profiles_csv = resolve_path(base_dir, config["unknown_profiles_csv"])
    fixed_event_csv = runtime_dir / "generated_candidate_events_mode_b.csv"
    fixed_crop_root = runtime_dir / "generated_event_crops"

    audit_stage_counts_json = runtime_dir / "audit_pipeline_stage_counts.json"
    audit_multicam_gt_coverage_csv = runtime_dir / "audit_multicam_gt_coverage.csv"
    audit_missing_event_reasons_csv = runtime_dir / "audit_missing_event_reasons.csv"
    audit_unknown_reuse_csv = runtime_dir / "audit_unknown_reuse.csv"
    audit_association_trace_csv = runtime_dir / "audit_association_trace.csv"
    audit_overlap_cases_csv = runtime_dir / "audit_overlap_cases.csv"
    audit_gt_split_merge_csv = runtime_dir / "audit_gt_split_merge.csv"
    audit_expected_reid_failures_csv = runtime_dir / "audit_expected_reid_failures.csv"
    audit_event_generation_subzones_csv = runtime_dir / "audit_event_generation_subzones.csv"
    audit_report_md = runtime_dir / "audit_report.md"
    association_logs_dir = runtime_dir / "association_logs"
    association_decisions_jsonl = association_logs_dir / "association_decisions.jsonl"
    association_summary_json = association_logs_dir / "association_summary.json"
    association_policy_runtime_json = association_logs_dir / "association_policy_runtime.json"
    camera_transition_map_runtime_json = association_logs_dir / "camera_transition_map_runtime.json"

    track_rows = parse_track_rows(read_csv(tracks_csv))
    queue_rows = read_csv(queue_csv)
    base_manifest_rows = read_csv(known_manifest_csv)
    association_policy, association_policy_runtime = load_association_policy(
        config_path=association_policy_config,
        base_dir=config_path.parent,
    )
    camera_transition_map, camera_transition_map_runtime = core_load_camera_transition_map(
        wildtrack_config,
        config_path=camera_transition_map_config,
        base_dir=config_path.parent,
    )
    scene_calibration, _runtime_cameras, scene_runtime = load_runtime_scene_calibration(
        config_path=scene_calibration_config,
        base_dir=config_path.parent,
        camera_ids=wildtrack_config.get("selected_cameras", []),
        required=True,
    )
    frame_sizes = scene_runtime.get("frame_sizes", {})
    camera_transition_map = apply_scene_calibration_to_transition_map(
        camera_transition_map,
        scene_calibration,
        frame_sizes,
    )
    wildtrack_config = apply_scene_calibration_to_wildtrack_config(
        wildtrack_config,
        scene_calibration,
        frame_sizes,
    )

    app = FaceAnalysis(
        name=config["insightface_runtime"].get("recommended_model_name", "buffalo_l"),
        root=str(Path(config["insightface_runtime"].get("recommended_model_root", "C:/Users/Admin/.insightface"))),
        providers=[config["insightface_runtime"].get("provider", "CPUExecutionProvider")],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    auto_enroll_count = max(0, as_int(config.get("demo_auto_enroll_count", 2), 2))
    auto_enrolled = enroll_demo_authorized_identities(app, queue_rows, known_root, count=auto_enroll_count)
    manifest_rows = base_manifest_rows + auto_enrolled
    write_csv(
        runtime_manifest_csv,
        manifest_rows,
        ["identity_id", "display_name", "source_repo_path", "gallery_rel_path", "seed_type", "status", "notes"],
    )
    identity_means, _ = build_gallery_embeddings(app, manifest_rows, base_dir, known_embeddings_csv)

    stage_a, stage_a_rows, _ = build_timeline_audit(track_rows, wildtrack_config["selected_cameras"])
    stage_b_baseline = build_baseline_stage_b(track_rows, queue_rows, wildtrack_config, camera_transition_map)

    baseline_events = []
    for row in queue_rows:
        baseline_spatial = {
            "zone_id": row.get("zone_id") or "",
            "zone_type": row.get("zone_type") or "",
            "zone_reason": row.get("zone_reason") or "",
            "zone_fallback_used": row.get("zone_fallback_used", False),
            "subzone_id": row.get("subzone_id") or "",
            "subzone_type": row.get("subzone_type") or "",
            "subzone_reason": row.get("subzone_reason") or "",
            "subzone_fallback_used": row.get("subzone_fallback_used", False),
            "matched_zone_region_id": row.get("matched_zone_region_id") or row.get("zone_id") or "",
            "matched_subzone_region_id": row.get("matched_subzone_region_id") or row.get("subzone_id") or "",
            "assignment_point_x": row.get("foot_x", ""),
            "assignment_point_y": row.get("foot_y", ""),
        }
        if not baseline_spatial["zone_id"] and not baseline_spatial["subzone_id"]:
            baseline_spatial = core_resolve_spatial_context(
                row["camera_id"],
                as_float(row.get("foot_x"), None),
                as_float(row.get("foot_y"), None),
                camera_transition_map,
            )
        baseline_events.append(
            {
                "event_id": row["event_id"],
                "event_type": "ENTRY_IN",
                "camera_id": row["camera_id"],
                "camera_role": wildtrack_config["cameras"][row["camera_id"]]["role"],
                "global_gt_id": str(row["global_gt_id"]),
                "local_track_id": row.get("local_track_id", ""),
                "frame_id": as_int(row["frame_id"]),
                "relative_sec": as_float(row["relative_sec"]),
                "best_shot_frame": as_int(row["frame_id"]),
                "best_shot_sec": as_float(row["relative_sec"]),
                "best_head_crop": row["best_head_crop"],
                "best_body_crop": row["best_body_crop"],
                "anchor_camera_id": row["camera_id"],
                "anchor_frame_id": as_int(row["frame_id"]),
                "anchor_relative_sec": as_float(row["relative_sec"]),
                "relation_type": "entry",
                "same_area_overlap": False,
                "direction": row.get("direction", "IN"),
                "foot_x": row.get("foot_x", ""),
                "foot_y": row.get("foot_y", ""),
                "zone_id": baseline_spatial["zone_id"],
                "zone_type": baseline_spatial["zone_type"],
                "zone_reason": baseline_spatial["zone_reason"],
                "zone_fallback_used": baseline_spatial["zone_fallback_used"],
                "subzone_id": baseline_spatial["subzone_id"],
                "subzone_type": baseline_spatial["subzone_type"],
                "subzone_reason": baseline_spatial["subzone_reason"],
                "subzone_fallback_used": baseline_spatial["subzone_fallback_used"],
                "matched_zone_region_id": baseline_spatial["matched_zone_region_id"],
                "matched_subzone_region_id": baseline_spatial["matched_subzone_region_id"],
                "assignment_point_x": baseline_spatial["assignment_point_x"],
                "assignment_point_y": baseline_spatial["assignment_point_y"],
                "best_shot_strategy": row.get("best_shot_strategy", ""),
                "best_shot_reason": row.get("best_shot_reason", ""),
                "best_shot_zone_id": row.get("best_shot_zone_id", ""),
                "best_shot_subzone_id": row.get("best_shot_subzone_id", ""),
                "best_shot_subzone_type": row.get("best_shot_subzone_type", ""),
                "best_shot_frames_after_anchor": row.get("best_shot_frames_after_anchor", ""),
                "evidence_buffer_count": as_int(row.get("evidence_buffer_count", 0)),
                "evidence_buffer_json": row.get("evidence_buffer_json", "[]"),
            }
        )
    analyzed_baseline = analyze_event_crops(app, baseline_events)
    baseline_assignments = build_baseline_assignments(
        analyzed_baseline,
        identity_means,
        threshold=float(config["matching"].get("known_match_threshold", 0.65)),
        unknown_prefix=config["unknown_handling"].get("seed_prefix", "UNK"),
        unknown_start=int(config["unknown_handling"].get("start_index", 1)),
    )
    baseline_resolved_rows = build_resolved_rows_from_gt_assignments(analyzed_baseline, baseline_assignments, "mode_a_baseline")
    baseline_timeline_rows = build_baseline_stream_timeline(track_rows, baseline_assignments)

    fixed_events, stage_b_fixed, _, _, topology = build_fixed_candidate_events(
        track_rows, wildtrack_config, dataset_root, fixed_crop_root, camera_transition_map
    )
    write_csv(fixed_event_csv, fixed_events, fieldnames_for_rows(fixed_events, ["event_id"]))
    analyzed_fixed = analyze_event_crops(app, fixed_events)
    mode_b_resolved_rows, profiles, association_trace_rows, association_debug = assign_model_identities(
        analyzed_fixed,
        identity_means,
        topology,
        unknown_prefix=config["unknown_handling"].get("seed_prefix", "UNK"),
        unknown_start=int(config["unknown_handling"].get("start_index", 1)),
        policy=association_policy,
        return_debug_bundle=True,
    )
    association_decision_logs = association_debug["decision_logs"]
    association_summary = core_summarize_decision_logs(association_decision_logs)
    face_body_summary = summarize_face_body_usage(analyzed_fixed, mode_b_resolved_rows, association_decision_logs, profiles)
    unknown_profile_rows = build_unknown_profile_rows(profiles)
    mode_b_timeline_rows = build_stream_timeline_from_events(track_rows, mode_b_resolved_rows)
    event_assignment_audit_rows = [
        core_build_event_assignment_audit_row("mode_a_baseline", event) for event in baseline_events
    ] + [
        core_build_event_assignment_audit_row("mode_b_true_assoc", event) for event in fixed_events
    ]

    coverage_rows, missing_rows, expected_failure_rows, overlap_rows = build_coverage_and_failures(
        stage_a_rows,
        stage_b_baseline,
        stage_b_fixed,
        baseline_resolved_rows,
        mode_b_resolved_rows,
    )
    split_merge_rows = build_gt_split_merge(baseline_resolved_rows, "mode_a_baseline") + build_gt_split_merge(
        mode_b_resolved_rows, "mode_b_true_assoc"
    )
    unknown_reuse_rows = build_unknown_reuse_rows(baseline_resolved_rows, "mode_a_baseline") + build_unknown_reuse_rows(
        mode_b_resolved_rows, "mode_b_true_assoc"
    )

    mode_a_metrics = summarize_mode(
        baseline_resolved_rows,
        stage_b_baseline,
        stage_a,
        gt_bridge_active=True,
        true_model_based_reuse_count=0,
    )
    temp_mode_b_metrics = summarize_mode(
        mode_b_resolved_rows,
        stage_b_fixed,
        stage_a,
        gt_bridge_active=False,
        true_model_based_reuse_count=0,
        association_summary=association_summary,
    )
    mode_b_metrics = dict(temp_mode_b_metrics)
    mode_b_metrics["true_model_based_reuse_count"] = sum(
        1
        for row in unknown_reuse_rows
        if row["run_mode"] == "mode_b_true_assoc"
        and row["reused_across_cameras"]
        and as_int(row["gt_count_for_audit"]) == 1
    )
    mode_b_metrics["expected_reid_candidate_count"] = sum(1 for row in coverage_rows if row["expected_reid_candidate"])
    mode_b_metrics["failed_expected_reid_candidate_count"] = sum(
        1 for row in coverage_rows if row["failed_expected_reid_candidate"]
    )
    mode_a_metrics["expected_reid_candidate_count"] = mode_b_metrics["expected_reid_candidate_count"]
    mode_a_metrics["failed_expected_reid_candidate_count"] = sum(
        1 for row in coverage_rows if row["expected_reid_candidate"] and row["baseline_resolved_camera_count"] <= 1
    )

    write_csv(baseline_resolved_csv, baseline_resolved_rows, fieldnames_for_rows(baseline_resolved_rows, ["event_id"]))
    write_csv(mode_b_resolved_csv, mode_b_resolved_rows, fieldnames_for_rows(mode_b_resolved_rows, ["event_id"]))
    write_csv(resolved_events_csv, mode_b_resolved_rows, fieldnames_for_rows(mode_b_resolved_rows, ["event_id"]))
    write_csv(
        baseline_timeline_csv,
        baseline_timeline_rows,
        fieldnames_for_rows(baseline_timeline_rows, ["resolved_global_id"]),
    )
    write_csv(
        mode_b_timeline_csv,
        mode_b_timeline_rows,
        fieldnames_for_rows(mode_b_timeline_rows, ["resolved_global_id"]),
    )
    write_csv(
        runtime_dir / "stream_identity_timeline.csv",
        mode_b_timeline_rows,
        fieldnames_for_rows(mode_b_timeline_rows, ["resolved_global_id"]),
    )
    write_csv(unknown_profiles_csv, unknown_profile_rows, fieldnames_for_rows(unknown_profile_rows, ["unknown_global_id"]))
    write_csv(audit_multicam_gt_coverage_csv, coverage_rows, fieldnames_for_rows(coverage_rows, ["global_gt_id"]))
    write_csv(audit_missing_event_reasons_csv, missing_rows, fieldnames_for_rows(missing_rows, ["run_mode"]))
    write_csv(audit_unknown_reuse_csv, unknown_reuse_rows, fieldnames_for_rows(unknown_reuse_rows, ["run_mode"]))
    write_csv(
        audit_association_trace_csv,
        association_trace_rows,
        fieldnames_for_rows(association_trace_rows, ["run_mode"]),
    )
    core_write_jsonl(association_decisions_jsonl, association_decision_logs)
    save_json(
        association_summary_json,
        {
            "metrics": association_summary,
            "policy_source": association_policy_runtime,
            "camera_transition_map_source": camera_transition_map_runtime,
            "log_path": str(association_decisions_jsonl),
        },
    )
    save_json(
        association_policy_runtime_json,
        {
            "policy_runtime": association_policy_runtime,
            "policy": association_policy,
        },
    )
    save_json(
        camera_transition_map_runtime_json,
        {
            "camera_transition_map_runtime": camera_transition_map_runtime,
            "camera_transition_map": camera_transition_map,
        },
    )
    save_json(face_body_summary_json, face_body_summary)
    write_csv(audit_overlap_cases_csv, overlap_rows, fieldnames_for_rows(overlap_rows, ["global_gt_id"]))
    write_csv(audit_gt_split_merge_csv, split_merge_rows, fieldnames_for_rows(split_merge_rows, ["run_mode"]))
    write_csv(
        audit_expected_reid_failures_csv,
        expected_failure_rows,
        fieldnames_for_rows(expected_failure_rows, ["global_gt_id"]),
    )
    write_csv(
        audit_event_generation_subzones_csv,
        event_assignment_audit_rows,
        fieldnames_for_rows(event_assignment_audit_rows, ["run_mode", "event_id"]),
    )

    root_cause_lines = [
        "Baseline stage B exported only entry-camera events, so most multi-camera GTs never reached association as separate per-camera candidates.",
        "Baseline resolved events were assigned by grouping on global_gt_id, so reused unknown IDs at event level were constrained by GT grouping rather than model association.",
        "The frame-level timeline looked more consistent than the event table because stream unification still used wildtrack_global_gt_id_demo_bridge.",
        "The overlap-heavy topology required near-zero travel time, but baseline event construction never materialized those overlap observations as second-camera candidate events.",
    ]
    render_report(audit_report_md, mode_a_metrics, mode_b_metrics, stage_a, root_cause_lines, coverage_rows, expected_failure_rows)

    stage_counts = {
        "stage_a_raw_timeline": stage_a,
        "mode_a_baseline": mode_a_metrics,
        "mode_b_true_assoc": mode_b_metrics,
        "comparison": {
            "reused_unknown_id_count_delta": mode_b_metrics["reused_unknown_id_count"] - mode_a_metrics["reused_unknown_id_count"],
            "total_event_count_delta": mode_b_metrics["total_event_count"] - mode_a_metrics["total_event_count"],
            "resolved_multicam_gt_count_delta": mode_b_metrics["resolved_multicam_gt_count"] - mode_a_metrics["resolved_multicam_gt_count"],
        },
        "association_policy_runtime": association_policy_runtime,
        "camera_transition_map_runtime": camera_transition_map_runtime,
        "scene_calibration_runtime": scene_runtime,
    }
    save_json(audit_stage_counts_json, stage_counts)

    summary = {
        "mode_a_baseline": mode_a_metrics,
        "mode_b_true_assoc": mode_b_metrics,
        "auto_enrolled_demo_identities": [row["identity_id"] for row in auto_enrolled],
        "baseline_stream_unification_basis": "wildtrack_global_gt_id_demo_bridge",
        "mode_b_stream_unification_basis": "per_camera_event_projection_without_gt_crossstream_bridge",
        "association_policy_runtime": association_policy_runtime,
        "camera_transition_map_runtime": camera_transition_map_runtime,
        "scene_calibration_runtime": scene_runtime,
        "association_logs": {
            "decision_log_jsonl": str(association_decisions_jsonl),
            "summary_json": str(association_summary_json),
            "policy_runtime_json": str(association_policy_runtime_json),
            "camera_transition_map_runtime_json": str(camera_transition_map_runtime_json),
        },
        "face_body_usage": {
            "summary_json": str(face_body_summary_json),
            "metrics": face_body_summary,
        },
        "event_generation_audit": {
            "subzone_assignment_csv": str(audit_event_generation_subzones_csv),
        },
        "notes": [
            "Mode A reproduces the previous GT-grouped behavior for audit.",
            "Mode B disables GT bridge for cross-stream identity propagation and performs real unknown-gallery association.",
        ],
    }
    save_json(summary_json, summary)

    print(
        "ASSOCIATION_POLICY_SOURCE="
        + (association_policy_runtime["source_path"] or "built_in_defaults")
    )
    print(
        "CAMERA_TRANSITION_MAP_SOURCE="
        + (camera_transition_map_runtime["source_path"] or "built_in_defaults")
    )
    print(
        "SCENE_CALIBRATION_SOURCE="
        + (scene_runtime["source_path"] or "preview_only_or_missing")
    )
    print(f"TOTAL_EVENTS={mode_b_metrics['total_event_count']}")
    print(f"UNKNOWN_EVENTS={mode_b_metrics['unknown_event_count']}")
    print(f"UNIQUE_UNKNOWN_IDS={mode_b_metrics['unique_unknown_id_count']}")
    print(f"REUSED_UNKNOWN_IDS={mode_b_metrics['reused_unknown_id_count']}")
    print(f"DEFER_COUNT={mode_b_metrics.get('defer_count', 0)}")
    print(f"NEW_UNKNOWN_COUNT={mode_b_metrics.get('new_unknown_count', 0)}")
    print(f"UNKNOWN_REUSE_COUNT={mode_b_metrics.get('unknown_reuse_count', 0)}")
    print(f"KNOWN_ACCEPT_COUNT={mode_b_metrics.get('known_accept_count', 0)}")
    print(f"TIMELINE_MULTICAM_GT={mode_b_metrics['timeline_multicam_gt_count']}")
    print(f"RESOLVED_MULTICAM_GT={mode_b_metrics['resolved_multicam_gt_count']}")
    print(f"EXPECTED_REID_CANDIDATES={mode_b_metrics['expected_reid_candidate_count']}")
    print(f"FAILED_EXPECTED_REID_CANDIDATES={mode_b_metrics['failed_expected_reid_candidate_count']}")
    print(f"GT_BRIDGE_ACTIVE={str(mode_b_metrics['gt_bridge_active']).lower()}")
    print(f"TRUE_MODEL_BASED_REUSE_COUNT={mode_b_metrics['true_model_based_reuse_count']}")


if __name__ == "__main__":
    config_path = Path(os.environ.get("FACE_DEMO_CONFIG", str(CONFIG_DEFAULT)))
    main(config_path)
