import copy


SHAPE_TYPE_PRESETS = {
    "processing_roi": {
        "label": "Processing ROI",
        "requires_shape_id": False,
        "requires_polygon": True,
        "kind_options": [],
    },
    "entry_line": {
        "label": "Entry Line",
        "requires_shape_id": False,
        "requires_polygon": False,
        "kind_options": [],
    },
    "zone": {
        "label": "Zone",
        "requires_shape_id": True,
        "requires_polygon": True,
        "kind_options": ["entry", "exit", "interior", "overlap", "transit", "main"],
    },
    "subzone": {
        "label": "Subzone",
        "requires_shape_id": True,
        "requires_polygon": True,
        "kind_options": ["entry", "exit", "interior", "overlap", "transit", "staging"],
    },
}


def _copy_camera_cfg(camera_cfg):
    cfg = copy.deepcopy(camera_cfg or {})
    cfg.setdefault("processing_roi", {"polygon": []})
    cfg.setdefault("entry_line", {"points": [], "in_side_point": []})
    cfg.setdefault("zones", [])
    cfg.setdefault("subzones", [])
    cfg.setdefault("default_zone_id", "")
    cfg.setdefault("default_subzone_id", "")
    return cfg


def undo_last_draft_point(draft_points):
    if not draft_points:
        return []
    return list(draft_points[:-1])


def build_shape_catalog(camera_cfg):
    cfg = _copy_camera_cfg(camera_cfg)
    shapes = []
    processing_polygon = (cfg.get("processing_roi", {}) or {}).get("polygon", []) or []
    if processing_polygon:
        shapes.append(
            {
                "shape_type": "processing_roi",
                "shape_id": "processing_roi",
                "label": "Processing ROI",
                "point_count": len(processing_polygon),
                "kind_value": "",
                "selected": False,
            }
        )
    entry_line = (cfg.get("entry_line", {}) or {}).get("points", []) or []
    if len(entry_line) == 2:
        shapes.append(
            {
                "shape_type": "entry_line",
                "shape_id": "entry_line",
                "label": "Entry Line",
                "point_count": 3 if (cfg.get("entry_line", {}) or {}).get("in_side_point") else 2,
                "kind_value": "",
                "selected": False,
            }
        )
    for zone in cfg.get("zones", []) or []:
        shapes.append(
            {
                "shape_type": "zone",
                "shape_id": zone.get("zone_id", ""),
                "label": zone.get("zone_id", ""),
                "point_count": len(zone.get("polygon", []) or []),
                "kind_value": zone.get("zone_type", ""),
                "selected": False,
            }
        )
    for subzone in cfg.get("subzones", []) or []:
        shapes.append(
            {
                "shape_type": "subzone",
                "shape_id": subzone.get("subzone_id", ""),
                "label": subzone.get("subzone_id", ""),
                "point_count": len(subzone.get("polygon", []) or []),
                "kind_value": subzone.get("subzone_type", ""),
                "parent_zone_id": subzone.get("parent_zone_id", ""),
                "selected": False,
            }
        )
    return shapes


def commit_draft_shape(
    camera_cfg,
    *,
    shape_type,
    draft_points,
    shape_id="",
    kind_value="",
    parent_zone_id="",
    priority=100,
    description="",
):
    cfg = _copy_camera_cfg(camera_cfg)
    points = copy.deepcopy(draft_points or [])
    if shape_type not in SHAPE_TYPE_PRESETS:
        raise ValueError(f"unsupported shape_type: {shape_type}")
    if shape_type == "processing_roi":
        if len(points) < 3:
            raise ValueError("processing_roi requires at least 3 points")
        cfg["processing_roi"] = {"polygon": points}
        return cfg
    if shape_type == "entry_line":
        if len(points) < 3:
            raise ValueError("entry_line requires 3 points: p1, p2, in-side")
        cfg["entry_line"] = {"points": points[:2], "in_side_point": points[2]}
        return cfg
    if len(points) < 3:
        raise ValueError(f"{shape_type} requires at least 3 points")
    if not shape_id:
        raise ValueError(f"{shape_type} requires a shape_id")
    if shape_type == "zone":
        cfg["zones"] = [item for item in (cfg.get("zones", []) or []) if item.get("zone_id") != shape_id]
        cfg["zones"].append(
            {
                "zone_id": shape_id,
                "zone_type": kind_value,
                "polygon": points,
                "priority": int(priority),
                "description": description,
                "placeholder": False,
            }
        )
        if not cfg.get("default_zone_id"):
            cfg["default_zone_id"] = shape_id
        return cfg
    cfg["subzones"] = [item for item in (cfg.get("subzones", []) or []) if item.get("subzone_id") != shape_id]
    cfg["subzones"].append(
        {
            "subzone_id": shape_id,
            "parent_zone_id": parent_zone_id or cfg.get("default_zone_id", ""),
            "subzone_type": kind_value,
            "polygon": points,
            "priority": int(priority),
            "description": description,
            "placeholder": False,
            "allowed_transitions": [],
        }
    )
    if not cfg.get("default_subzone_id"):
        cfg["default_subzone_id"] = shape_id
    return cfg


def delete_shape(camera_cfg, *, shape_type, shape_id=""):
    cfg = _copy_camera_cfg(camera_cfg)
    if shape_type == "processing_roi":
        cfg["processing_roi"] = {"polygon": []}
        return cfg
    if shape_type == "entry_line":
        cfg["entry_line"] = {"points": [], "in_side_point": []}
        return cfg
    if shape_type == "zone":
        cfg["zones"] = [item for item in (cfg.get("zones", []) or []) if item.get("zone_id") != shape_id]
        if cfg.get("default_zone_id") == shape_id:
            cfg["default_zone_id"] = (cfg["zones"][0] or {}).get("zone_id", "") if cfg["zones"] else ""
        cfg["subzones"] = [
            item for item in (cfg.get("subzones", []) or []) if item.get("parent_zone_id") != shape_id
        ]
        if cfg.get("default_subzone_id") and not any(
            item.get("subzone_id") == cfg["default_subzone_id"] for item in cfg.get("subzones", []) or []
        ):
            cfg["default_subzone_id"] = (cfg["subzones"][0] or {}).get("subzone_id", "") if cfg["subzones"] else ""
        return cfg
    if shape_type == "subzone":
        cfg["subzones"] = [item for item in (cfg.get("subzones", []) or []) if item.get("subzone_id") != shape_id]
        if cfg.get("default_subzone_id") == shape_id:
            cfg["default_subzone_id"] = (cfg["subzones"][0] or {}).get("subzone_id", "") if cfg["subzones"] else ""
        return cfg
    raise ValueError(f"unsupported shape_type: {shape_type}")
