import copy
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_SCENE_CALIBRATION = {
    "version": 1,
    "coordinate_space": "normalized",
    "processing": {
        "require_manual_calibration": True,
        "roi_mask_enabled": True,
        "mask_outside_polygon": True,
        "preview_overlay_enabled": True,
        "preview_interval_sec": 1.0,
    },
    "direction_filter": {
        "history_window": 6,
        "minimum_points": 4,
        "minimum_inward_motion_px": 18.0,
        "minimum_inside_ratio": 0.55,
        "require_zone_transition": False,
        "allow_line_cross_only_when_history_short": False,
        "inward_zone_types": ["exit", "interior", "overlap", "transit"],
        "outward_zone_types": ["entry", "outer", "approach"],
    },
    "cameras": {},
}


def resolve_path(base_dir: Path, value: str | None):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _candidate_paths(config_path=None):
    runtime_dir = Path(__file__).resolve().parent
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(runtime_dir / "config" / "manual_scene_calibration.json")
    candidates.append(runtime_dir / "config" / "manual_scene_calibration.wildtrack.json")
    return candidates


def _load_payload(path: Path):
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _deep_merge(base, updates):
    merged = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_camera(camera_id, camera_cfg):
    normalized = copy.deepcopy(camera_cfg or {})
    normalized["camera_id"] = normalized.get("camera_id", camera_id)
    normalized.setdefault("role", "")
    normalized.setdefault("description", "")
    normalized.setdefault("preview_source", "")
    normalized.setdefault("preview_source_type", "file")
    normalized.setdefault("frame_size_ref", {"width": 1920, "height": 1080})
    normalized.setdefault("processing_roi", {"polygon": []})
    normalized.setdefault("entry_line", {"points": [], "in_side_point": []})
    normalized.setdefault("default_zone_id", "")
    normalized.setdefault("default_subzone_id", "")
    normalized.setdefault("entry_zones", [])
    normalized.setdefault("exit_zones", [])
    zones = []
    for index, zone in enumerate(normalized.get("zones", []) or []):
        item = copy.deepcopy(zone)
        item.setdefault("zone_id", f"{camera_id.lower()}_zone_{index:02d}")
        item.setdefault("zone_type", "")
        item.setdefault("polygon", [])
        item.setdefault("priority", 0)
        item.setdefault("description", "")
        item.setdefault("placeholder", False)
        zones.append(item)
    subzones = []
    for index, subzone in enumerate(normalized.get("subzones", []) or []):
        item = copy.deepcopy(subzone)
        item.setdefault("subzone_id", f"{camera_id.lower()}_subzone_{index:02d}")
        item.setdefault("parent_zone_id", normalized.get("default_zone_id", ""))
        item.setdefault("subzone_type", "")
        item.setdefault("polygon", [])
        item.setdefault("priority", 0)
        item.setdefault("description", "")
        item.setdefault("placeholder", False)
        item.setdefault("allowed_transitions", [])
        subzones.append(item)
    normalized["zones"] = zones
    normalized["subzones"] = subzones
    return normalized


def build_blank_scene_calibration(camera_ids=None):
    calibration = copy.deepcopy(DEFAULT_SCENE_CALIBRATION)
    for camera_id in camera_ids or []:
        calibration["cameras"][camera_id] = _normalize_camera(camera_id, {})
    return calibration


def build_blank_camera_calibration(camera_id, template=None):
    base = {
        "camera_id": camera_id,
        "role": (template or {}).get("role", ""),
        "description": (template or {}).get("description", ""),
        "preview_source": (template or {}).get("preview_source", ""),
        "preview_source_type": (template or {}).get("preview_source_type", "file"),
        "frame_size_ref": copy.deepcopy((template or {}).get("frame_size_ref", {"width": 1920, "height": 1080})),
    }
    return _normalize_camera(camera_id, base)


def _validate_normalized_points(points, minimum_points):
    if len(points) < minimum_points:
        return False
    for point in points:
        if len(point) != 2:
            return False
        x, y = point
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            return False
    return True


def validate_scene_calibration(calibration):
    errors = []
    warnings = []
    if calibration.get("coordinate_space") != "normalized":
        errors.append("coordinate_space must be 'normalized'")
    for camera_id, camera_cfg in (calibration.get("cameras", {}) or {}).items():
        role = camera_cfg.get("role", "")
        frame_size_ref = camera_cfg.get("frame_size_ref", {}) or {}
        if float(frame_size_ref.get("width", 0) or 0) <= 0 or float(frame_size_ref.get("height", 0) or 0) <= 0:
            errors.append(f"{camera_id}: frame_size_ref.width and frame_size_ref.height must be > 0")
        processing_roi = (camera_cfg.get("processing_roi", {}) or {}).get("polygon", []) or []
        if not _validate_normalized_points(processing_roi, 3):
            errors.append(f"{camera_id}: processing_roi.polygon must contain at least 3 normalized points")
        if role == "entry":
            entry_line = camera_cfg.get("entry_line", {}) or {}
            if not _validate_normalized_points(entry_line.get("points", []) or [], 2):
                errors.append(f"{camera_id}: entry_line.points must contain exactly 2 normalized points")
            inside_point = entry_line.get("in_side_point", []) or []
            if not _validate_normalized_points([inside_point], 1):
                errors.append(f"{camera_id}: entry_line.in_side_point must be a normalized point")
        if not camera_cfg.get("zones"):
            warnings.append(f"{camera_id}: no zones configured")
        if not camera_cfg.get("subzones"):
            warnings.append(f"{camera_id}: no subzones configured")
    return errors, warnings


def load_scene_calibration(config_path=None, base_dir=None, required=True, camera_ids=None):
    checked_paths = []
    source_path = None
    loaded = None
    for candidate in _candidate_paths(config_path):
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute() and base_dir:
            candidate_path = (Path(base_dir) / candidate_path).resolve()
        checked_paths.append(str(candidate_path))
        if not candidate_path.exists():
            continue
        payload = _load_payload(candidate_path)
        loaded = payload.get("scene_calibration", payload) if isinstance(payload, dict) else {}
        source_path = candidate_path
        break

    if loaded is None:
        if required:
            raise RuntimeError(
                "Manual scene calibration config is required for runtime. "
                "Deprecated auto/inferred ROI fallback is disabled. "
                f"Checked: {checked_paths}"
            )
        calibration = build_blank_scene_calibration(camera_ids=camera_ids)
        errors, warnings = validate_scene_calibration(calibration)
        return calibration, {
            "source_path": "",
            "checked_paths": checked_paths,
            "preview_only": True,
            "errors": errors,
            "warnings": warnings,
            "camera_count": len(calibration.get("cameras", {})),
        }

    calibration = _deep_merge(DEFAULT_SCENE_CALIBRATION, loaded)
    cameras = {}
    selected_ids = camera_ids or list((calibration.get("cameras", {}) or {}).keys())
    for camera_id in selected_ids:
        cameras[camera_id] = _normalize_camera(camera_id, (calibration.get("cameras", {}) or {}).get(camera_id, {}))
    calibration["cameras"] = cameras
    errors, warnings = validate_scene_calibration(calibration)
    if errors and required:
        raise RuntimeError("Invalid manual scene calibration config: " + "; ".join(errors))
    runtime_info = {
        "source_path": str(source_path) if source_path else "",
        "checked_paths": checked_paths,
        "preview_only": False,
        "errors": errors,
        "warnings": warnings,
        "camera_count": len(cameras),
        "coordinate_space": calibration.get("coordinate_space", ""),
    }
    return calibration, runtime_info


def save_scene_calibration(path: Path, calibration):
    payload = {"scene_calibration": calibration}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _point_to_pixels(point, frame_width, frame_height):
    return [int(round(float(point[0]) * frame_width)), int(round(float(point[1]) * frame_height))]


def _polygon_to_pixels(polygon, frame_width, frame_height):
    return [_point_to_pixels(point, frame_width, frame_height) for point in polygon or []]


def build_runtime_camera_calibration(camera_cfg, frame_width, frame_height):
    runtime_camera = copy.deepcopy(camera_cfg)
    runtime_camera["frame_width"] = int(frame_width)
    runtime_camera["frame_height"] = int(frame_height)
    runtime_camera["processing_roi"] = _polygon_to_pixels(
        (camera_cfg.get("processing_roi", {}) or {}).get("polygon", []),
        frame_width,
        frame_height,
    )
    entry_line = camera_cfg.get("entry_line", {}) or {}
    runtime_camera["entry_line"] = _polygon_to_pixels(entry_line.get("points", []), frame_width, frame_height)
    inside_point = entry_line.get("in_side_point", []) or []
    runtime_camera["in_side_point"] = (
        _point_to_pixels(inside_point, frame_width, frame_height) if len(inside_point) == 2 else []
    )
    runtime_camera["zones"] = []
    for zone in camera_cfg.get("zones", []) or []:
        item = copy.deepcopy(zone)
        item["polygon"] = _polygon_to_pixels(zone.get("polygon", []), frame_width, frame_height)
        runtime_camera["zones"].append(item)
    runtime_camera["subzones"] = []
    for subzone in camera_cfg.get("subzones", []) or []:
        item = copy.deepcopy(subzone)
        item["polygon"] = _polygon_to_pixels(subzone.get("polygon", []), frame_width, frame_height)
        runtime_camera["subzones"].append(item)
    return runtime_camera


def merge_camera_geometry_into_transition_map(transition_map, camera_id, runtime_camera):
    merged = copy.deepcopy(transition_map)
    merged.setdefault("cameras", {})
    existing = copy.deepcopy((merged.get("cameras", {}) or {}).get(camera_id, {}))
    existing.update(
        {
            "camera_id": camera_id,
            "role": runtime_camera.get("role", existing.get("role", "")),
            "description": runtime_camera.get("description", existing.get("description", "")),
            "default_zone_id": runtime_camera.get("default_zone_id", existing.get("default_zone_id", "")),
            "default_subzone_id": runtime_camera.get("default_subzone_id", existing.get("default_subzone_id", "")),
            "entry_zones": copy.deepcopy(runtime_camera.get("entry_zones", existing.get("entry_zones", []))),
            "exit_zones": copy.deepcopy(runtime_camera.get("exit_zones", existing.get("exit_zones", []))),
            "zones": copy.deepcopy(runtime_camera.get("zones", [])),
            "subzones": copy.deepcopy(runtime_camera.get("subzones", [])),
            "processing_roi": copy.deepcopy(runtime_camera.get("processing_roi", [])),
            "entry_line": copy.deepcopy(runtime_camera.get("entry_line", [])),
            "in_side_point": copy.deepcopy(runtime_camera.get("in_side_point", [])),
            "manual_calibration_active": True,
        }
    )
    merged["cameras"][camera_id] = existing
    merged.setdefault("runtime_policy", {})
    merged["runtime_policy"]["disallow_default_region_fallback"] = True
    merged["runtime_policy"]["manual_scene_calibration_required"] = True
    return merged


def apply_scene_calibration_to_transition_map(transition_map, calibration, frame_sizes, require_complete=True):
    merged = copy.deepcopy(transition_map)
    for camera_id, camera_cfg in (calibration.get("cameras", {}) or {}).items():
        frame_size = frame_sizes.get(camera_id)
        if frame_size is None:
            if require_complete:
                raise RuntimeError(f"Missing frame size for calibrated camera {camera_id}")
            frame_size = camera_cfg.get("frame_size_ref", {}) or {}
        runtime_camera = build_runtime_camera_calibration(
            camera_cfg,
            int(frame_size.get("width", 1920)),
            int(frame_size.get("height", 1080)),
        )
        merged = merge_camera_geometry_into_transition_map(merged, camera_id, runtime_camera)
    merged.setdefault("runtime_policy", {})
    merged["runtime_policy"]["manual_scene_calibration_source"] = calibration.get("source_path", "")
    merged["runtime_policy"]["scene_processing"] = copy.deepcopy(calibration.get("processing", {}) or {})
    merged["runtime_policy"]["direction_filter"] = copy.deepcopy(calibration.get("direction_filter", {}) or {})
    return merged


def apply_scene_calibration_to_wildtrack_config(wildtrack_config, calibration, frame_sizes, require_complete=True):
    merged = copy.deepcopy(wildtrack_config)
    merged["direction_filter"] = copy.deepcopy(calibration.get("direction_filter", {}) or {})
    merged["processing"] = copy.deepcopy(calibration.get("processing", {}) or {})
    for camera_id in merged.get("selected_cameras", []):
        if camera_id not in calibration.get("cameras", {}):
            if require_complete:
                raise RuntimeError(f"Missing manual scene calibration for camera {camera_id}")
            continue
        camera_cfg = calibration["cameras"][camera_id]
        frame_size = frame_sizes.get(camera_id)
        if frame_size is None:
            if require_complete:
                raise RuntimeError(f"Missing frame size for calibrated camera {camera_id}")
            frame_size = camera_cfg.get("frame_size_ref", {}) or {}
        runtime_camera = build_runtime_camera_calibration(
            camera_cfg,
            int(frame_size.get("width", 1920)),
            int(frame_size.get("height", 1080)),
        )
        merged_camera = copy.deepcopy(merged["cameras"][camera_id])
        merged_camera["manual_calibration_active"] = True
        merged_camera["direction_filter"] = copy.deepcopy(calibration.get("direction_filter", {}) or {})
        merged_camera["processing_roi"] = runtime_camera.get("processing_roi", [])
        if runtime_camera.get("role") == "entry":
            merged_camera["entry_roi"] = runtime_camera.get("processing_roi", [])
            merged_camera["entry_line"] = runtime_camera.get("entry_line", [])
            merged_camera["in_side_point"] = runtime_camera.get("in_side_point", [])
        else:
            merged_camera["track_roi"] = runtime_camera.get("processing_roi", [])
        merged_camera["zones"] = copy.deepcopy(runtime_camera.get("zones", []))
        merged_camera["subzones"] = copy.deepcopy(runtime_camera.get("subzones", []))
        merged["cameras"][camera_id] = merged_camera
    merged["scene_calibration_required"] = True
    return merged


def build_frame_sizes(camera_ids, calibration, source_probe=None):
    frame_sizes = {}
    for camera_id in camera_ids:
        if source_probe:
            probed = source_probe(camera_id)
            if probed:
                frame_sizes[camera_id] = probed
                continue
        camera_cfg = (calibration.get("cameras", {}) or {}).get(camera_id, {})
        ref = camera_cfg.get("frame_size_ref", {}) or {}
        if float(ref.get("width", 0) or 0) > 0 and float(ref.get("height", 0) or 0) > 0:
            frame_sizes[camera_id] = {"width": int(ref["width"]), "height": int(ref["height"])}
    return frame_sizes


def source_probe_factory(calibration, base_dir: Path | None = None, source_lookup=None):
    source_lookup = source_lookup or {}
    base_dir = Path(base_dir or Path.cwd())

    def probe(camera_id):
        source_cfg = source_lookup.get(camera_id, {}) or {}
        source_type = source_cfg.get("source_type")
        source_value = source_cfg.get("source")
        if not source_type or source_value in ("", None):
            camera_cfg = (calibration.get("cameras", {}) or {}).get(camera_id, {})
            source_type = camera_cfg.get("preview_source_type", "file")
            source_value = camera_cfg.get("preview_source", "")
        if not source_value:
            return None
        resolved = resolve_path(base_dir, str(source_value))
        try:
            return probe_frame_size_from_source(source_type, str(resolved))
        except Exception:
            return None

    return probe


def load_runtime_scene_calibration(
    config_path=None,
    base_dir=None,
    camera_ids=None,
    source_lookup=None,
    required=True,
):
    calibration, runtime_info = load_scene_calibration(
        config_path=config_path,
        base_dir=base_dir,
        required=required,
        camera_ids=camera_ids,
    )
    frame_sizes = build_frame_sizes(
        camera_ids or list((calibration.get("cameras", {}) or {}).keys()),
        calibration,
        source_probe=source_probe_factory(calibration, base_dir=base_dir, source_lookup=source_lookup),
    )
    runtime_cameras = {}
    for camera_id, camera_cfg in (calibration.get("cameras", {}) or {}).items():
        frame_size = frame_sizes.get(camera_id) or (camera_cfg.get("frame_size_ref", {}) or {})
        runtime_cameras[camera_id] = build_runtime_camera_calibration(
            camera_cfg,
            int(frame_size.get("width", 1920)),
            int(frame_size.get("height", 1080)),
        )
    runtime_info = copy.deepcopy(runtime_info)
    runtime_info["frame_sizes"] = frame_sizes
    return calibration, runtime_cameras, runtime_info


def _probe_frame_from_image(image_path: Path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"Failed to read preview image: {image_path}")
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to decode preview image: {image_path}")
    return frame


def probe_frame_from_source(source_type, source_value, frame_idx=0):
    if source_type == "image":
        return _probe_frame_from_image(Path(source_value))
    capture = cv2.VideoCapture(int(source_value) if source_type == "webcam" else str(source_value))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open preview source: {source_value}")
    try:
        if source_type in {"file", "rtsp"} and frame_idx:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to decode preview frame from {source_value}")
        return frame
    finally:
        capture.release()


def probe_frame_size_from_source(source_type, source_value):
    frame = probe_frame_from_source(source_type, source_value, frame_idx=0)
    height, width = frame.shape[:2]
    return {"width": int(width), "height": int(height)}


def build_processing_mask(frame_shape, runtime_camera):
    height, width = frame_shape[:2]
    polygon = runtime_camera.get("processing_roi", []) or []
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    else:
        mask[:] = 255
    return mask


def apply_processing_mask(frame, runtime_camera, enabled=True):
    if not enabled:
        return frame
    mask = build_processing_mask(frame.shape, runtime_camera)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked


def draw_scene_overlay(frame, runtime_camera, track_boxes=None):
    overlay = frame.copy()
    processing_roi = runtime_camera.get("processing_roi", []) or []
    if len(processing_roi) >= 3:
        cv2.polylines(overlay, [np.asarray(processing_roi, dtype=np.int32)], True, (24, 140, 255), 2)
        cv2.putText(overlay, "processing_roi", tuple(processing_roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (24, 140, 255), 2)

    for zone in runtime_camera.get("zones", []) or []:
        polygon = np.asarray(zone.get("polygon", []), dtype=np.int32)
        if polygon.size == 0:
            continue
        cv2.polylines(overlay, [polygon], True, (20, 110, 60), 2)
        x, y = polygon[0].tolist()
        cv2.putText(overlay, zone.get("zone_id", "zone"), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 110, 60), 2)

    for subzone in runtime_camera.get("subzones", []) or []:
        polygon = np.asarray(subzone.get("polygon", []), dtype=np.int32)
        if polygon.size == 0:
            continue
        cv2.polylines(overlay, [polygon], True, (190, 90, 30), 2)
        x, y = polygon[0].tolist()
        cv2.putText(overlay, subzone.get("subzone_id", "subzone"), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 90, 30), 1)

    if runtime_camera.get("entry_line"):
        p1, p2 = runtime_camera["entry_line"]
        cv2.line(overlay, tuple(map(int, p1)), tuple(map(int, p2)), (220, 60, 140), 2)
        inside = runtime_camera.get("in_side_point", [])
        if inside:
            cv2.circle(overlay, tuple(map(int, inside)), 6, (220, 60, 140), -1)
            cv2.putText(overlay, "IN side", tuple(map(int, inside)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 60, 140), 1)

    for box in track_boxes or []:
        cv2.rectangle(
            overlay,
            (int(box["xmin"]), int(box["ymin"])),
            (int(box["xmax"]), int(box["ymax"])),
            (245, 245, 245),
            2,
        )
        cv2.putText(
            overlay,
            str(box.get("label", "")),
            (int(box["xmin"]), max(20, int(box["ymin"]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 245, 245),
            1,
        )
    return overlay
