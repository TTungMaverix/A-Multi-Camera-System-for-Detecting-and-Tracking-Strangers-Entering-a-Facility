from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_ASSOCIATION_POLICY = {
    "quality_gate": {
        "min_bbox_area": 1200.0,
        "min_body_area": 1800.0,
        "full_body_area": 22000.0,
        "reliable_face_det_score": 0.20,
        "strong_face_det_score": 0.35,
        "min_face_bbox_width": 36,
        "min_face_bbox_height": 36,
        "min_face_bbox_area": 1296,
        "min_face_blur_score": 45.0,
        "max_abs_yaw_deg": 30.0,
        "max_abs_pitch_deg": 20.0,
        "max_abs_roll_deg": 20.0,
        "require_landmarks_for_face_buffer": True,
        "max_face_buffer_candidates_per_track": 3,
    },
    "topology_filter": {
        "relation_priors": {
            "overlap": 1.0,
            "sequential": 0.85,
            "weak_link": 0.65,
        },
        "overlap": {
            "max_time_floor_sec": 0.5,
        },
        "sequential": {
            "min_window_span_sec": 0.5,
        },
        "weak_link": {
            "fallback_max_travel_time_sec": 2.0,
            "require_non_negative_delta": True,
        },
        "zone": {
            "default_allow_when_missing": True,
        },
        "subzone": {
            "default_allow_when_missing": True,
        },
    },
    "appearance_evidence": {
        "prefer_primary_face_when_reliable": True,
        "allow_body_primary_fallback": True,
        "enable_face_body_fusion": True,
        "fusion_face_weight": 0.7,
        "fusion_body_weight": 0.3,
        "secondary_available_min_score": 0.0,
        "face_reference_weight": 0.7,
        "face_representative_weight": 0.3,
        "body_reference_weight": 0.65,
        "body_representative_weight": 0.35,
    },
    "body_reid": {
        "enabled": True,
        "extractor_name": "osnet_x0_25",
        "device": "cpu",
        "pretrained": True,
        "input_width": 128,
        "input_height": 256,
        "min_crop_width": 12,
        "min_crop_height": 24,
        "clahe_enabled": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": 8,
        "gray_world_normalization": False,
        "tracklet_pooling_max_candidates": 10,
        "tracklet_pooling_top_k": 5,
        "tracklet_pooling_min_blur_score": 25.0,
        "tracklet_pooling_min_bbox_area": 1800.0,
        "tracklet_pooling_min_relative_bbox_area": 0.55,
    },
    "gallery_lifecycle": {
        "top_k_face_refs": 3,
        "top_k_body_refs": 5,
        "ttl_sec": 12.0,
        "face_quality_area_bonus": 0.001,
        "representative_update_policy": "mean_top_k_refs",
        "expiry_policy": "hard_ttl",
    },
    "decision_policy": {
        "known_accept_threshold": 0.65,
        "known_margin_threshold": 0.02,
        "unknown_reuse_threshold": 0.18,
        "create_rule": "create_new_unknown_when_no_safe_match",
        "pending_policy": {
            "enabled": True,
            "window_frames": 30,
            "window_sec": 1.0,
            "max_buffer_items": 8,
            "stale_timeout_sec": 2.0,
        },
        "defer_policy": {
            "quality_reliability_max": 0.35,
            "quality_gate_fail_action": "defer",
            "ambiguous_low_quality_action": "create_unknown",
        },
        "margin_by_relation": {
            "overlap": 0.015,
            "sequential": 0.02,
            "weak_link": 0.04,
            "camera_already_seen": 1.0,
            "unreachable": 1.0,
            "no_link": 1.0,
        },
        "relation_thresholds": {
            "overlap": {
                "fusion_primary": 0.18,
                "face_primary": 0.18,
                "face_secondary": 0.30,
                "body_primary": 0.60,
                "body_secondary": 0.18,
            },
            "sequential": {
                "fusion_primary": 0.20,
                "face_primary": 0.20,
                "face_secondary": 0.35,
                "body_primary": 0.58,
                "body_secondary": 0.18,
            },
            "weak_link": {
                "fusion_primary": 0.24,
                "face_primary": 0.24,
                "face_secondary": 0.40,
                "body_primary": 0.64,
                "body_secondary": 0.22,
            },
        },
        "minimum_evidence": {
            "require_secondary_when_available": True,
        },
        "topology_supported_accept": {
            "enabled": False,
            "allowed_relations": ["sequential"],
            "require_primary_modality": "body",
            "min_time_score": 0.9,
            "max_primary_shortfall": 0.1,
            "require_exact_zone": True,
            "require_exact_subzone": True,
        },
    },
}


def deep_merge(base, updates):
    merged = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
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


def _candidate_policy_paths(config_path):
    runtime_dir = Path(__file__).resolve().parent.parent
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(runtime_dir / "config" / "association_policy.yaml")
    candidates.append(runtime_dir / "config" / "association_policy.example.yaml")
    return candidates


def load_association_policy(config_path=None, base_dir=None):
    checked_paths = []
    loaded = {}
    source_path = None
    for candidate in _candidate_policy_paths(config_path):
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute() and base_dir:
            candidate_path = (Path(base_dir) / candidate_path).resolve()
        checked_paths.append(str(candidate_path))
        if not candidate_path.exists():
            continue
        payload = yaml.safe_load(candidate_path.read_text(encoding="utf-8")) or {}
        loaded = payload.get("association_policy", payload) if isinstance(payload, dict) else {}
        source_path = candidate_path
        break

    merged = deep_merge(DEFAULT_ASSOCIATION_POLICY, loaded)
    runtime_info = {
        "source_path": str(source_path) if source_path else "",
        "used_builtin_defaults_only": source_path is None,
        "checked_paths": checked_paths,
        "defaulted_keys": _flatten_missing_keys(DEFAULT_ASSOCIATION_POLICY, loaded),
    }
    return merged, runtime_info
