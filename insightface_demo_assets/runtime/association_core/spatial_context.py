def point_in_polygon(x, y, polygon):
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


def _camera_cfg(camera_id, transition_map):
    return (transition_map.get("cameras", {}) or {}).get(camera_id, {})


def _fallback_disabled(transition_map):
    runtime_policy = transition_map.get("runtime_policy", {}) or {}
    return bool(runtime_policy.get("disallow_default_region_fallback", False))


def _sorted_regions(regions):
    return sorted(
        regions or [],
        key=lambda item: (
            int(item.get("priority", 0)),
            len(item.get("polygon", []) or []),
            item.get("zone_id", ""),
            item.get("subzone_id", ""),
        ),
        reverse=True,
    )


def default_zone_for_camera(camera_id, transition_map):
    if _fallback_disabled(transition_map):
        return {
            "zone_id": "",
            "zone_type": "",
            "matched_zone_region_id": "",
            "zone_reason": "manual_zone_required",
            "zone_fallback_used": False,
        }
    camera_cfg = _camera_cfg(camera_id, transition_map)
    default_zone_id = camera_cfg.get("default_zone_id", "")
    zones = camera_cfg.get("zones", []) or []
    selected_zone = None
    if default_zone_id:
        selected_zone = next((zone for zone in zones if zone.get("zone_id") == default_zone_id), None)
    if selected_zone is None and zones:
        selected_zone = _sorted_regions(zones)[0]
    zone_id = selected_zone.get("zone_id", default_zone_id) if selected_zone else default_zone_id
    zone_type = selected_zone.get("zone_type", "") if selected_zone else ""
    return {
        "zone_id": zone_id or "",
        "zone_type": zone_type,
        "matched_zone_region_id": zone_id or "",
        "zone_reason": "camera_default_zone" if zone_id else "zone_unavailable",
        "zone_fallback_used": bool(zone_id),
    }


def default_subzone_for_camera(camera_id, transition_map, zone_id=""):
    if _fallback_disabled(transition_map):
        return {
            "subzone_id": "",
            "subzone_type": "",
            "matched_subzone_region_id": "",
            "subzone_reason": "manual_subzone_required",
            "subzone_fallback_used": False,
        }
    camera_cfg = _camera_cfg(camera_id, transition_map)
    default_subzone_id = camera_cfg.get("default_subzone_id", "")
    subzones = camera_cfg.get("subzones", []) or []
    selected = None
    if default_subzone_id:
        selected = next((subzone for subzone in subzones if subzone.get("subzone_id") == default_subzone_id), None)
    if selected is None and zone_id:
        selected = next(
            (
                subzone
                for subzone in _sorted_regions(subzones)
                if subzone.get("parent_zone_id", "") in {"", zone_id}
            ),
            None,
        )
    if selected is None and subzones:
        selected = _sorted_regions(subzones)[0]
    subzone_id = selected.get("subzone_id", default_subzone_id) if selected else default_subzone_id
    subzone_type = selected.get("subzone_type", "") if selected else ""
    return {
        "subzone_id": subzone_id or "",
        "subzone_type": subzone_type,
        "matched_subzone_region_id": subzone_id or "",
        "subzone_reason": "camera_default_subzone" if subzone_id else "subzone_unavailable",
        "subzone_fallback_used": bool(subzone_id),
    }


def _resolve_zone_for_point(camera_id, point_x, point_y, transition_map):
    camera_cfg = _camera_cfg(camera_id, transition_map)
    zones = camera_cfg.get("zones", []) or []
    if point_x is None or point_y is None:
        return default_zone_for_camera(camera_id, transition_map)
    for zone in _sorted_regions(zones):
        polygon = zone.get("polygon", []) or []
        if polygon and point_in_polygon(float(point_x), float(point_y), polygon):
            zone_id = zone.get("zone_id", "")
            return {
                "zone_id": zone_id,
                "zone_type": zone.get("zone_type", ""),
                "matched_zone_region_id": zone_id,
                "zone_reason": "point_in_config_zone",
                "zone_fallback_used": False,
            }
    if _fallback_disabled(transition_map):
        return {
            "zone_id": "",
            "zone_type": "",
            "matched_zone_region_id": "",
            "zone_reason": "point_outside_manual_zone",
            "zone_fallback_used": False,
        }
    fallback = default_zone_for_camera(camera_id, transition_map)
    if fallback["zone_id"]:
        fallback["zone_reason"] = "default_zone_outside_polygon"
    return fallback


def _resolve_subzone_for_point(camera_id, point_x, point_y, transition_map, zone_id=""):
    camera_cfg = _camera_cfg(camera_id, transition_map)
    subzones = camera_cfg.get("subzones", []) or []
    if point_x is None or point_y is None:
        return default_subzone_for_camera(camera_id, transition_map, zone_id=zone_id)
    filtered = [
        subzone
        for subzone in subzones
        if not zone_id or subzone.get("parent_zone_id", "") in {"", zone_id}
    ]
    for subzone in _sorted_regions(filtered):
        polygon = subzone.get("polygon", []) or []
        if polygon and point_in_polygon(float(point_x), float(point_y), polygon):
            subzone_id = subzone.get("subzone_id", "")
            return {
                "subzone_id": subzone_id,
                "subzone_type": subzone.get("subzone_type", ""),
                "matched_subzone_region_id": subzone_id,
                "subzone_reason": "point_in_config_subzone",
                "subzone_fallback_used": False,
            }
    if _fallback_disabled(transition_map):
        return {
            "subzone_id": "",
            "subzone_type": "",
            "matched_subzone_region_id": "",
            "subzone_reason": "point_outside_manual_subzone",
            "subzone_fallback_used": False,
        }
    fallback = default_subzone_for_camera(camera_id, transition_map, zone_id=zone_id)
    if fallback["subzone_id"]:
        fallback["subzone_reason"] = "default_subzone_outside_polygon"
    return fallback


def resolve_spatial_context(camera_id, point_x, point_y, transition_map):
    zone_meta = _resolve_zone_for_point(camera_id, point_x, point_y, transition_map)
    subzone_meta = _resolve_subzone_for_point(camera_id, point_x, point_y, transition_map, zone_id=zone_meta["zone_id"])
    return {
        **zone_meta,
        **subzone_meta,
        "assignment_point_x": point_x if point_x is not None else "",
        "assignment_point_y": point_y if point_y is not None else "",
    }


def build_event_assignment_audit_row(run_mode, event):
    return {
        "run_mode": run_mode,
        "event_id": event.get("event_id", ""),
        "event_type": event.get("event_type", ""),
        "camera_id": event.get("camera_id", ""),
        "relative_sec": event.get("relative_sec", ""),
        "zone_id": event.get("zone_id", ""),
        "zone_type": event.get("zone_type", ""),
        "subzone_id": event.get("subzone_id", ""),
        "subzone_type": event.get("subzone_type", ""),
        "assignment_point_x": event.get("assignment_point_x", event.get("foot_x", "")),
        "assignment_point_y": event.get("assignment_point_y", event.get("foot_y", "")),
        "matched_zone_region_id": event.get("matched_zone_region_id", event.get("zone_id", "")),
        "matched_subzone_region_id": event.get("matched_subzone_region_id", event.get("subzone_id", "")),
        "zone_reason": event.get("zone_reason", ""),
        "subzone_reason": event.get("subzone_reason", ""),
        "zone_fallback_used": event.get("zone_fallback_used", False),
        "subzone_fallback_used": event.get("subzone_fallback_used", False),
        "best_shot_strategy": event.get("best_shot_strategy", ""),
        "best_shot_reason": event.get("best_shot_reason", ""),
        "best_shot_zone_id": event.get("best_shot_zone_id", ""),
        "best_shot_subzone_id": event.get("best_shot_subzone_id", ""),
        "best_shot_subzone_type": event.get("best_shot_subzone_type", ""),
        "best_shot_frames_after_anchor": event.get("best_shot_frames_after_anchor", ""),
        "direction_reason": event.get("direction_reason", ""),
        "direction_history_points": event.get("direction_history_points", ""),
        "direction_momentum_px": event.get("direction_momentum_px", ""),
        "direction_inside_ratio": event.get("direction_inside_ratio", ""),
    }
