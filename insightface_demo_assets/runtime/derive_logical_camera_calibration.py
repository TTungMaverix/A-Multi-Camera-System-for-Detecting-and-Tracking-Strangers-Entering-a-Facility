import argparse
import copy
from pathlib import Path

from scene_calibration import load_scene_calibration, save_scene_calibration, validate_scene_calibration


DEFAULT_DERIVATIONS = [("C1", "C3"), ("C2", "C4")]


def _camera_text_map(derivations=None):
    text_map = {}
    for src_camera_id, dst_camera_id in derivations or DEFAULT_DERIVATIONS:
        text_map[src_camera_id.upper()] = dst_camera_id.upper()
        text_map[src_camera_id.lower()] = dst_camera_id.lower()
    return text_map


def _rename_scoped_text(value, text_map):
    if not isinstance(value, str):
        return value
    updated = value
    for before, after in text_map.items():
        updated = updated.replace(before, after)
    return updated


def _rename_zone_payload(item, text_map):
    payload = copy.deepcopy(item or {})
    for key in [
        "zone_id",
        "subzone_id",
        "parent_zone_id",
        "description",
    ]:
        payload[key] = _rename_scoped_text(payload.get(key, ""), text_map)
    payload["allowed_transitions"] = [
        _rename_scoped_text(value, text_map)
        for value in (payload.get("allowed_transitions", []) or [])
    ]
    return payload


def derive_logical_camera_entry(camera_cfg, src_camera_id, dst_camera_id, *, text_map=None):
    derived = copy.deepcopy(camera_cfg or {})
    text_map = text_map or _camera_text_map()
    derived["camera_id"] = dst_camera_id
    derived["description"] = f"Logical replay of {src_camera_id}."
    derived["default_zone_id"] = _rename_scoped_text(derived.get("default_zone_id", ""), text_map)
    derived["default_subzone_id"] = _rename_scoped_text(
        derived.get("default_subzone_id", ""),
        text_map,
    )
    derived["entry_zones"] = [
        _rename_scoped_text(value, text_map)
        for value in (derived.get("entry_zones", []) or [])
    ]
    derived["exit_zones"] = [
        _rename_scoped_text(value, text_map)
        for value in (derived.get("exit_zones", []) or [])
    ]
    derived["zones"] = [
        _rename_zone_payload(item, text_map)
        for item in (derived.get("zones", []) or [])
    ]
    derived["subzones"] = [
        _rename_zone_payload(item, text_map)
        for item in (derived.get("subzones", []) or [])
    ]
    return derived


def derive_logical_cameras(calibration, derivations=None):
    next_calibration = copy.deepcopy(calibration or {})
    next_calibration.setdefault("cameras", {})
    text_map = _camera_text_map(derivations)
    for src_camera_id, dst_camera_id in derivations or DEFAULT_DERIVATIONS:
        src_cfg = (next_calibration.get("cameras", {}) or {}).get(src_camera_id)
        if not src_cfg:
            raise RuntimeError(f"Missing source camera calibration: {src_camera_id}")
        next_calibration["cameras"][dst_camera_id] = derive_logical_camera_entry(
            src_cfg,
            src_camera_id,
            dst_camera_id,
            text_map=text_map,
        )
    return next_calibration


def parse_args():
    parser = argparse.ArgumentParser(description="Derive logical replay camera calibration from physical cameras.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to manual scene calibration config to update in place.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    calibration, _runtime = load_scene_calibration(
        config_path=str(config_path),
        base_dir=config_path.parent,
        required=True,
    )
    updated = derive_logical_cameras(calibration)
    errors, warnings = validate_scene_calibration(updated)
    if errors:
        raise RuntimeError("Derived calibration is invalid: " + "; ".join(errors))
    save_scene_calibration(config_path, updated)
    print(f"DERIVED_LOGICAL_CAMERAS={config_path}")
    if warnings:
        print("WARNINGS=" + " | ".join(warnings))


if __name__ == "__main__":
    main()
