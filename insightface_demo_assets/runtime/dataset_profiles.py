import copy
import json
from pathlib import Path

import yaml


DEFAULT_DATASET_PROFILE = {
    "profile_name": "",
    "dataset_name": "",
    "dataset_root": ".",
    "assumed_video_fps": 10.0,
    "selected_cameras": [],
    "best_shot_window_frames": 80,
    "line_crossing_distance_threshold": 140.0,
    "best_shot_selection": {},
    "head_crop": {
        "top_ratio": 0.02,
        "bottom_ratio": 0.45,
        "side_ratio": 0.18,
    },
    "face_buffer": {},
    "camera_topology": {},
    "cameras": {},
    "physical_cameras": {},
    "clip_pairing": {
        "naming_mode": "shared_stem",
        "supported_extensions": [".mp4", ".mov", ".avi", ".mkv"],
        "default_pair_id": "",
    },
    "logical_demo": {
        "enabled": False,
        "description": "",
        "logical_cameras": {},
    },
}


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
    normalized.setdefault("view_index", "")
    normalized.setdefault("preview_source", "")
    normalized.setdefault("preview_source_type", "file")
    normalized.setdefault("anchor_point_mode", "bottom_center")
    normalized.setdefault("source_physical_camera_id", "")
    normalized.setdefault("timeline_offset_sec", 0.0)
    normalized.setdefault("logical_demo_copy", False)
    return normalized


def _normalize_physical_camera(physical_camera_id, physical_cfg):
    normalized = copy.deepcopy(physical_cfg or {})
    normalized["physical_camera_id"] = normalized.get("physical_camera_id", physical_camera_id)
    normalized.setdefault("label", physical_camera_id)
    normalized.setdefault("root_dir", "")
    normalized.setdefault("description", "")
    return normalized


def _normalize_profile(profile, source_path: Path | None = None):
    normalized = _deep_merge(DEFAULT_DATASET_PROFILE, profile or {})
    if not normalized.get("profile_name"):
        normalized["profile_name"] = (source_path.stem if source_path else "dataset_profile")
    if not normalized.get("dataset_name"):
        normalized["dataset_name"] = normalized["profile_name"]
    normalized["assumed_video_fps"] = float(normalized.get("assumed_video_fps", 10.0) or 10.0)
    selected_cameras = list(normalized.get("selected_cameras", []) or [])
    cameras = {}
    for camera_id, camera_cfg in (normalized.get("cameras", {}) or {}).items():
        cameras[camera_id] = _normalize_camera(camera_id, camera_cfg)
    normalized["cameras"] = cameras
    if not selected_cameras:
        selected_cameras = list(cameras.keys())
    normalized["selected_cameras"] = selected_cameras
    normalized["physical_cameras"] = {
        physical_camera_id: _normalize_physical_camera(physical_camera_id, physical_cfg)
        for physical_camera_id, physical_cfg in (normalized.get("physical_cameras", {}) or {}).items()
    }
    logical_demo = copy.deepcopy(normalized.get("logical_demo", {}) or {})
    logical_demo.setdefault("enabled", False)
    logical_demo.setdefault("description", "")
    logical_demo.setdefault("logical_cameras", {})
    for camera_id in selected_cameras:
        logical_cfg = copy.deepcopy((logical_demo.get("logical_cameras", {}) or {}).get(camera_id, {}))
        logical_cfg.setdefault("logical_camera_id", camera_id)
        logical_cfg.setdefault("source_physical_camera_id", cameras.get(camera_id, {}).get("source_physical_camera_id", ""))
        logical_cfg.setdefault("timeline_offset_sec", float(cameras.get(camera_id, {}).get("timeline_offset_sec", 0.0) or 0.0))
        logical_cfg.setdefault("logical_demo_copy", bool(cameras.get(camera_id, {}).get("logical_demo_copy", False)))
        logical_demo["logical_cameras"][camera_id] = logical_cfg
    normalized["logical_demo"] = logical_demo
    return normalized


def resolve_dataset_profile_path(config, project_root: Path, base_dir: Path | None = None):
    candidate = config.get("dataset_profile_config") or config.get("wildtrack_demo_config") or ""
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        base_candidate = (Path(base_dir) / path).resolve()
        if base_candidate.exists():
            return base_candidate
    return (project_root / path).resolve()


def load_dataset_profile(config_path: Path):
    payload = _load_payload(config_path)
    profile = payload.get("dataset_profile", payload) if isinstance(payload, dict) else {}
    normalized = _normalize_profile(profile, source_path=config_path)
    runtime_info = {
        "source_path": str(config_path),
        "profile_name": normalized.get("profile_name", ""),
        "dataset_name": normalized.get("dataset_name", ""),
        "selected_cameras": list(normalized.get("selected_cameras", [])),
    }
    return normalized, runtime_info


def load_dataset_profile_from_config(config, project_root: Path, base_dir: Path | None = None):
    profile_path = resolve_dataset_profile_path(config, project_root, base_dir=base_dir)
    if profile_path is None:
        raise RuntimeError("Missing dataset_profile_config / wildtrack_demo_config in config.")
    profile, runtime_info = load_dataset_profile(profile_path)
    runtime_info["used_legacy_wildtrack_key"] = bool(config.get("wildtrack_demo_config")) and not bool(config.get("dataset_profile_config"))
    return profile_path, profile, runtime_info


def resolve_dataset_root(project_root: Path, config, dataset_profile):
    explicit_root = ((config.get("dataset", {}) or {}).get("root", "") or "").strip()
    dataset_root = explicit_root or str(dataset_profile.get("dataset_root", ".") or ".")
    path = Path(dataset_root)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _pair_extensions(clip_pairing_cfg):
    values = list(clip_pairing_cfg.get("supported_extensions", []) or [])
    if not values:
        values = [".mp4", ".mov", ".avi", ".mkv"]
    return {value.lower() for value in values}


def discover_clip_pairs(dataset_profile, project_root: Path):
    physical_cameras = dataset_profile.get("physical_cameras", {}) or {}
    if len(physical_cameras) < 2:
        raise RuntimeError("New dataset logical demo requires at least 2 physical camera roots in physical_cameras.")
    clip_pairing_cfg = dataset_profile.get("clip_pairing", {}) or {}
    allowed_extensions = _pair_extensions(clip_pairing_cfg)
    stems_by_camera = {}
    files_by_camera = {}
    for physical_camera_id, physical_cfg in physical_cameras.items():
        root_dir = physical_cfg.get("root_dir", "")
        if not root_dir:
            raise RuntimeError(f"physical_cameras.{physical_camera_id}.root_dir is required.")
        root_path = Path(root_dir)
        if not root_path.is_absolute():
            root_path = (project_root / root_path).resolve()
        if not root_path.exists():
            raise RuntimeError(f"Physical camera root does not exist: {root_path}")
        stem_map = {}
        for file_path in sorted(root_path.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_extensions:
                continue
            stem_map[file_path.stem] = file_path.resolve()
        stems_by_camera[physical_camera_id] = set(stem_map.keys())
        files_by_camera[physical_camera_id] = stem_map
    common_stems = None
    for stems in stems_by_camera.values():
        common_stems = stems if common_stems is None else (common_stems & stems)
    common_stems = sorted(common_stems or [])
    return {
        "available_pair_ids": common_stems,
        "files_by_camera": files_by_camera,
    }


def build_logical_demo_manifest(dataset_profile, project_root: Path, requested_pair_id: str | None = None):
    logical_demo_cfg = dataset_profile.get("logical_demo", {}) or {}
    if not logical_demo_cfg.get("enabled", False):
        raise RuntimeError("dataset_profile.logical_demo.enabled must be true for logical demo expansion.")
    discovered = discover_clip_pairs(dataset_profile, project_root)
    available_pair_ids = discovered["available_pair_ids"]
    if not available_pair_ids:
        raise RuntimeError("No clip pairs found across the configured physical camera folders.")
    clip_pairing_cfg = dataset_profile.get("clip_pairing", {}) or {}
    selected_pair_id = (
        requested_pair_id
        or clip_pairing_cfg.get("default_pair_id")
        or available_pair_ids[0]
    )
    if selected_pair_id not in available_pair_ids:
        raise RuntimeError(
            f"Requested pair_id={selected_pair_id!r} is not available. Found: {', '.join(available_pair_ids)}"
        )
    assumed_fps = float(dataset_profile.get("assumed_video_fps", 10.0) or 10.0)
    logical_rows = []
    video_sources = {}
    selected_cameras = list(dataset_profile.get("selected_cameras", []) or [])
    for camera_id in selected_cameras:
        logical_cfg = (logical_demo_cfg.get("logical_cameras", {}) or {}).get(camera_id, {})
        physical_camera_id = logical_cfg.get("source_physical_camera_id") or (dataset_profile.get("cameras", {}).get(camera_id, {}) or {}).get("source_physical_camera_id")
        if not physical_camera_id:
            raise RuntimeError(f"Missing source_physical_camera_id for logical camera {camera_id}.")
        physical_files = discovered["files_by_camera"].get(physical_camera_id, {})
        if selected_pair_id not in physical_files:
            raise RuntimeError(f"Pair {selected_pair_id!r} missing in physical camera {physical_camera_id}.")
        timeline_offset_sec = float(logical_cfg.get("timeline_offset_sec", 0.0) or 0.0)
        frame_offset = int(round(timeline_offset_sec * assumed_fps))
        source_path = physical_files[selected_pair_id]
        video_sources[camera_id] = str(source_path)
        logical_rows.append(
            {
                "logical_camera_id": camera_id,
                "physical_camera_id": physical_camera_id,
                "pair_id": selected_pair_id,
                "video_path": str(source_path),
                "timeline_offset_sec": round(timeline_offset_sec, 3),
                "frame_offset": frame_offset,
                "logical_demo_copy": bool(logical_cfg.get("logical_demo_copy", False)),
            }
        )
    physical_rows = []
    for physical_camera_id, file_map in discovered["files_by_camera"].items():
        selected = file_map.get(selected_pair_id)
        if selected is None:
            continue
        assigned_logical = [row["logical_camera_id"] for row in logical_rows if row["physical_camera_id"] == physical_camera_id]
        physical_rows.append(
            {
                "physical_camera_id": physical_camera_id,
                "pair_id": selected_pair_id,
                "video_path": str(selected),
                "assigned_logical_cameras": assigned_logical,
            }
        )
    return {
        "profile_name": dataset_profile.get("profile_name", ""),
        "dataset_name": dataset_profile.get("dataset_name", ""),
        "pair_id": selected_pair_id,
        "available_pair_ids": available_pair_ids,
        "video_sources": video_sources,
        "physical_sources": physical_rows,
        "logical_cameras": logical_rows,
        "notes": [
            "The current New Dataset has 2 physical cameras and is expanded into 4 logical demo streams by replaying the same paired clips with configured timeline offsets.",
            "Logical streams are explicit demo adapters for scope alignment, not claims of 4 independent physical cameras.",
        ],
    }


def materialize_profile_for_logical_demo(dataset_profile, manifest):
    materialized = copy.deepcopy(dataset_profile)
    logical_rows = {row["logical_camera_id"]: row for row in manifest.get("logical_cameras", [])}
    materialized["selected_cameras"] = [camera_id for camera_id in materialized.get("selected_cameras", []) if camera_id in logical_rows]
    for camera_id in materialized["selected_cameras"]:
        camera_cfg = materialized["cameras"][camera_id]
        logical_row = logical_rows[camera_id]
        camera_cfg["preview_source"] = logical_row["video_path"]
        camera_cfg["preview_source_type"] = "file"
        camera_cfg["source_physical_camera_id"] = logical_row["physical_camera_id"]
        camera_cfg["timeline_offset_sec"] = logical_row["timeline_offset_sec"]
        camera_cfg["logical_demo_copy"] = logical_row["logical_demo_copy"]
    return materialized
