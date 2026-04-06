from copy import deepcopy
from pathlib import Path

import yaml


def _deep_merge(base, updates):
    merged = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _flatten_missing_keys(defaults, loaded, prefix=""):
    missing = []
    for key, default_value in defaults.items():
        path = f"{prefix}.{key}" if prefix else key
        if not isinstance(loaded, dict) or key not in loaded:
            missing.append(path)
            continue
        loaded_value = loaded[key]
        if isinstance(default_value, dict) and isinstance(loaded_value, dict):
            missing.extend(_flatten_missing_keys(default_value, loaded_value, path))
    return missing


def _camera_zone_from_wildtrack(camera_id, camera_cfg):
    polygon = camera_cfg.get("entry_roi") or camera_cfg.get("track_roi") or []
    role = camera_cfg.get("role", "")
    default_zone_id = f"{camera_id.lower()}_{'entry' if role == 'entry' else 'track'}_default"
    zone_type = "entry" if role == "entry" else "transit"
    return {
        "camera_id": camera_id,
        "role": role,
        "description": camera_cfg.get("description", ""),
        "default_zone_id": default_zone_id,
        "entry_zones": [default_zone_id] if role == "entry" else [],
        "exit_zones": [default_zone_id],
        "zones": [
            {
                "zone_id": default_zone_id,
                "zone_type": zone_type,
                "polygon": polygon,
                "placeholder": False,
                "description": f"Derived from {'entry_roi' if camera_cfg.get('entry_roi') else 'track_roi'} in wildtrack_demo_config.json",
            }
        ],
    }


def build_default_transition_map(wildtrack_config):
    fps = float(wildtrack_config["assumed_video_fps"])
    cameras = {}
    for camera_id in wildtrack_config["selected_cameras"]:
        cameras[camera_id] = _camera_zone_from_wildtrack(camera_id, wildtrack_config["cameras"][camera_id])

    transitions = []
    for src_camera, targets in wildtrack_config["camera_topology"].items():
        for dst_camera, info in targets.items():
            src_default_zone = cameras.get(src_camera, {}).get("default_zone_id", "")
            dst_default_zone = cameras.get(dst_camera, {}).get("default_zone_id", "")
            min_sec = float(info.get("min_frame_gap", 0.0)) / fps
            max_sec = float(info.get("max_frame_gap", 0.0)) / fps
            transitions.append(
                {
                    "transition_id": f"{src_camera}_to_{dst_camera}_{info.get('relation', 'weak_link')}",
                    "src_camera_id": src_camera,
                    "dst_camera_id": dst_camera,
                    "relation_type": info.get("relation", "weak_link"),
                    "allowed_relation_types": [info.get("relation", "weak_link")],
                    "same_area_overlap": info.get("relation", "weak_link") == "overlap",
                    "min_travel_time_sec": min_sec,
                    "avg_travel_time_sec": (min_sec + max_sec) / 2.0,
                    "max_travel_time_sec": max_sec,
                    "allowed_exit_zones": [src_default_zone] if src_default_zone else [],
                    "allowed_entry_zones": [dst_default_zone] if dst_default_zone else [],
                    "placeholder": False,
                    "description": "Derived from camera_topology in wildtrack_demo_config.json",
                }
            )
    return {
        "cameras": cameras,
        "transitions": transitions,
    }


def _candidate_paths(config_path):
    runtime_dir = Path(__file__).resolve().parent.parent
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(runtime_dir / "config" / "camera_transition_map.yaml")
    candidates.append(runtime_dir / "config" / "camera_transition_map.example.yaml")
    return candidates


def load_camera_transition_map(wildtrack_config, config_path=None, base_dir=None):
    default_map = build_default_transition_map(wildtrack_config)
    checked_paths = []
    loaded = {}
    source_path = None
    for candidate in _candidate_paths(config_path):
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute() and base_dir:
            candidate_path = (Path(base_dir) / candidate_path).resolve()
        checked_paths.append(str(candidate_path))
        if not candidate_path.exists():
            continue
        payload = yaml.safe_load(candidate_path.read_text(encoding="utf-8")) or {}
        loaded = payload.get("camera_transition_map", payload) if isinstance(payload, dict) else {}
        source_path = candidate_path
        break

    merged = _deep_merge(default_map, loaded)
    runtime_info = {
        "source_path": str(source_path) if source_path else "",
        "used_builtin_defaults_only": source_path is None,
        "checked_paths": checked_paths,
        "defaulted_keys": _flatten_missing_keys(default_map, loaded),
        "camera_count": len(merged.get("cameras", {})),
        "transition_count": len(merged.get("transitions", [])),
    }
    return merged, runtime_info
