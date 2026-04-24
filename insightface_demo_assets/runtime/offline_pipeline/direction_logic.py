from copy import deepcopy


DEFAULT_DIRECTION_FILTER = {
    "history_window": 6,
    "minimum_points": 4,
    "minimum_inward_motion_px": 18.0,
    "minimum_inside_ratio": 0.55,
    "allow_late_start_inside_entry": True,
    "late_start_max_source_frame": 60,
    "late_start_min_inward_motion_px": 18.0,
    "late_start_min_inside_ratio": 0.9,
    "late_start_max_line_distance_ratio": 0.6,
    "late_start_max_line_distance_px": 240.0,
    "require_zone_transition": False,
    "allow_line_cross_only_when_history_short": False,
    "inward_zone_types": ["exit", "interior", "overlap", "transit"],
    "outward_zone_types": ["entry", "outer", "approach"],
}


def _cfg(config):
    merged = deepcopy(DEFAULT_DIRECTION_FILTER)
    for key, value in (config or {}).items():
        merged[key] = value
    return merged


def line_cross_value(point, line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    return ((x2 - x1) * (point["y"] - y1)) - ((y2 - y1) * (point["x"] - x1))


def _line_length(line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    return max((dx * dx + dy * dy) ** 0.5, 1e-6)


def is_in_side(point, line, in_side_point):
    anchor = {"x": float(in_side_point[0]), "y": float(in_side_point[1])}
    anchor_sign = line_cross_value(anchor, line)
    point_sign = line_cross_value(point, line)
    return (anchor_sign * point_sign) >= 0


def _signed_inward_value(point, line, in_side_point):
    anchor = {"x": float(in_side_point[0]), "y": float(in_side_point[1])}
    anchor_sign = line_cross_value(anchor, line)
    orientation = 1.0 if anchor_sign >= 0 else -1.0
    return orientation * (line_cross_value(point, line) / _line_length(line))


def _zone_transition(points_spatial, inward_zone_types, outward_zone_types):
    types = [
        (item.get("subzone_type") or item.get("zone_type") or "").lower()
        for item in points_spatial or []
    ]
    if not types:
        return False, "no_zone_history"
    seen_outward = any(zone_type in outward_zone_types for zone_type in types[:-1])
    ends_inward = any(zone_type in inward_zone_types for zone_type in types[-2:])
    if seen_outward and ends_inward:
        return True, "outward_to_inward_zone_transition"
    if ends_inward:
        return True, "ends_in_inward_zone_type"
    return False, "zone_transition_missing"


def evaluate_direction(points, line, in_side_point, spatial_history=None, config=None):
    cfg = _cfg(config)
    points = list(points or [])
    if len(points) > int(cfg["history_window"]):
        points = points[-int(cfg["history_window"]) :]
    spatial_history = list(spatial_history or [])
    if len(spatial_history) > len(points):
        spatial_history = spatial_history[-len(points) :]

    if len(points) < 2 or not line or not in_side_point:
        return {
            "decision": "NONE",
            "history_points": len(points),
            "cross_in": False,
            "cross_out": False,
            "momentum_px": 0.0,
            "inside_ratio": 0.0,
            "zone_transition_ok": False,
            "reason": "insufficient_geometry",
        }

    inside_flags = [is_in_side(point, line, in_side_point) for point in points]
    cross_in = any((not inside_flags[index - 1]) and inside_flags[index] for index in range(1, len(inside_flags)))
    cross_out = any(inside_flags[index - 1] and (not inside_flags[index]) for index in range(1, len(inside_flags)))
    signed_values = [_signed_inward_value(point, line, in_side_point) for point in points]
    first_window = max(1, min(3, len(points) // 2))
    last_window = max(1, min(3, len(points) // 2))
    start_mean = sum(signed_values[:first_window]) / float(first_window)
    end_mean = sum(signed_values[-last_window:]) / float(last_window)
    momentum_px = end_mean - start_mean
    inside_ratio = sum(1 for flag in inside_flags[-last_window:] if flag) / float(last_window)
    zone_transition_ok, zone_transition_reason = _zone_transition(
        spatial_history,
        inward_zone_types={zone.lower() for zone in cfg.get("inward_zone_types", []) or []},
        outward_zone_types={zone.lower() for zone in cfg.get("outward_zone_types", []) or []},
    )

    min_points = int(cfg["minimum_points"])
    min_inward_motion = float(cfg["minimum_inward_motion_px"])
    min_inside_ratio = float(cfg["minimum_inside_ratio"])
    if len(points) < min_points and not bool(cfg.get("allow_line_cross_only_when_history_short", False)):
        return {
            "decision": "NONE",
            "history_points": len(points),
            "cross_in": cross_in,
            "cross_out": cross_out,
            "momentum_px": round(momentum_px, 3),
            "inside_ratio": round(inside_ratio, 3),
            "zone_transition_ok": zone_transition_ok,
            "reason": "history_window_too_short",
        }

    require_zone_transition = bool(cfg.get("require_zone_transition", False))
    zone_ok = zone_transition_ok or (not require_zone_transition)
    if cross_in and inside_flags[-1] and momentum_px >= min_inward_motion and inside_ratio >= min_inside_ratio and zone_ok:
        return {
            "decision": "IN",
            "history_points": len(points),
            "cross_in": True,
            "cross_out": cross_out,
            "momentum_px": round(momentum_px, 3),
            "inside_ratio": round(inside_ratio, 3),
            "zone_transition_ok": zone_transition_ok,
            "reason": f"cross_in;momentum={round(momentum_px, 3)};inside_ratio={round(inside_ratio, 3)};zone={zone_transition_reason}",
        }
    if cross_out and (not inside_flags[-1]) and momentum_px <= (-min_inward_motion) and (1.0 - inside_ratio) >= min_inside_ratio:
        return {
            "decision": "OUT",
            "history_points": len(points),
            "cross_in": cross_in,
            "cross_out": True,
            "momentum_px": round(momentum_px, 3),
            "inside_ratio": round(inside_ratio, 3),
            "zone_transition_ok": zone_transition_ok,
            "reason": f"cross_out;momentum={round(momentum_px, 3)};inside_ratio={round(inside_ratio, 3)}",
        }
    return {
        "decision": "NONE",
        "history_points": len(points),
        "cross_in": cross_in,
        "cross_out": cross_out,
        "momentum_px": round(momentum_px, 3),
        "inside_ratio": round(inside_ratio, 3),
        "zone_transition_ok": zone_transition_ok,
        "reason": f"not_stable_enough;zone={zone_transition_reason}",
    }
