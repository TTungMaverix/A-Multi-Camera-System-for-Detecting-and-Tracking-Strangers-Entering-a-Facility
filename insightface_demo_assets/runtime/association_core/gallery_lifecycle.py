import numpy as np

from .config_loader import DEFAULT_ASSOCIATION_POLICY, deep_merge


def _merge_policy(policy):
    return deep_merge(DEFAULT_ASSOCIATION_POLICY["gallery_lifecycle"], policy or {})


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def keep_top_refs(refs, new_ref, top_k):
    refs.append(new_ref)
    refs.sort(key=lambda item: (item["quality_score"], item["relative_sec"]), reverse=True)
    del refs[top_k:]


def _representative_embedding(refs):
    vectors = [np.asarray(ref["embedding"], dtype=np.float32) for ref in refs if ref.get("embedding") is not None]
    if not vectors:
        return None
    stacked = np.stack(vectors, axis=0)
    return _normalize(np.mean(stacked, axis=0))


def _refresh_representatives(profile):
    profile["representative_face_embedding"] = _representative_embedding(profile.get("face_refs", []))
    profile["representative_body_embedding"] = _representative_embedding(profile.get("body_refs", []))
    profile["quality_stats"] = {
        "best_face_quality": round(max((ref["quality_score"] for ref in profile.get("face_refs", [])), default=0.0), 4),
        "best_body_quality": round(max((ref["quality_score"] for ref in profile.get("body_refs", [])), default=0.0), 4),
        "face_ref_count": len(profile.get("face_refs", [])),
        "body_ref_count": len(profile.get("body_refs", [])),
    }


def create_unknown_profile(unknown_global_id, item, policy=None):
    cfg = _merge_policy(policy)
    event = item["event"]
    profile = {
        "unknown_global_id": unknown_global_id,
        "first_seen_camera": event["camera_id"],
        "first_seen_time": float(event["relative_sec"]),
        "first_seen_zone": event.get("zone_id", ""),
        "first_seen_zone_type": event.get("zone_type", ""),
        "first_seen_subzone": event.get("subzone_id", ""),
        "first_seen_subzone_type": event.get("subzone_type", ""),
        "latest_seen_camera": event["camera_id"],
        "latest_seen_time": float(event["relative_sec"]),
        "latest_seen_zone": event.get("zone_id", ""),
        "latest_seen_zone_type": event.get("zone_type", ""),
        "latest_seen_subzone": event.get("subzone_id", ""),
        "latest_seen_subzone_type": event.get("subzone_type", ""),
        "history_cameras": [event["camera_id"]],
        "cameras_seen": [event["camera_id"]],
        "zones_seen": [event["zone_id"]] if event.get("zone_id") else [],
        "subzones_seen": [event["subzone_id"]] if event.get("subzone_id") else [],
        "event_ids": [event["event_id"]],
        "gt_ids": [event["global_gt_id"]],
        "face_refs": [],
        "body_refs": [],
        "top_k_face": int(cfg["top_k_face_refs"]),
        "top_k_body": int(cfg["top_k_body_refs"]),
        "ttl_sec": float(cfg["ttl_sec"]),
        "expiry_at_sec": float(event["relative_sec"]) + float(cfg["ttl_sec"]),
        "representative_face_embedding": None,
        "representative_body_embedding": None,
        "quality_stats": {},
    }
    update_unknown_profile(profile, item, policy=cfg)
    return profile


def update_unknown_profile(profile, item, policy=None):
    cfg = _merge_policy(policy)
    event = item["event"]
    profile["latest_seen_camera"] = event["camera_id"]
    profile["latest_seen_time"] = float(event["relative_sec"])
    profile["latest_seen_zone"] = event.get("zone_id", "")
    profile["latest_seen_zone_type"] = event.get("zone_type", "")
    profile["latest_seen_subzone"] = event.get("subzone_id", "")
    profile["latest_seen_subzone_type"] = event.get("subzone_type", "")
    profile["expiry_at_sec"] = float(event["relative_sec"]) + float(cfg["ttl_sec"])
    if event["camera_id"] not in profile["history_cameras"]:
        profile["history_cameras"].append(event["camera_id"])
    if event["camera_id"] not in profile["cameras_seen"]:
        profile["cameras_seen"].append(event["camera_id"])
    if event.get("zone_id") and event["zone_id"] not in profile["zones_seen"]:
        profile["zones_seen"].append(event["zone_id"])
    if event.get("subzone_id") and event["subzone_id"] not in profile["subzones_seen"]:
        profile["subzones_seen"].append(event["subzone_id"])
    if event["event_id"] not in profile["event_ids"]:
        profile["event_ids"].append(event["event_id"])
    if event["global_gt_id"] not in profile["gt_ids"]:
        profile["gt_ids"].append(event["global_gt_id"])
    if item.get("face_embedding") is not None:
        keep_top_refs(
            profile["face_refs"],
            {
                "embedding": item["face_embedding"],
                "event_id": event["event_id"],
                "camera_id": event["camera_id"],
                "relative_sec": float(event["relative_sec"]),
                "zone_id": event.get("zone_id", ""),
                "zone_type": event.get("zone_type", ""),
                "subzone_id": event.get("subzone_id", ""),
                "subzone_type": event.get("subzone_type", ""),
                "quality_score": float(item.get("face_det_score") or 0.0)
                + (float(cfg["face_quality_area_bonus"]) * float(event.get("bbox_area") or 0.0)),
                "crop_path": item.get("used_face_crop_path", ""),
            },
            profile["top_k_face"],
        )
    if item.get("body_embedding") is not None:
        keep_top_refs(
            profile["body_refs"],
            {
                "embedding": item["body_embedding"],
                "event_id": event["event_id"],
                "camera_id": event["camera_id"],
                "relative_sec": float(event["relative_sec"]),
                "zone_id": event.get("zone_id", ""),
                "zone_type": event.get("zone_type", ""),
                "subzone_id": event.get("subzone_id", ""),
                "subzone_type": event.get("subzone_type", ""),
                "quality_score": float(event.get("bbox_area") or 0.0),
                "crop_path": event.get("best_body_crop", ""),
            },
            profile["top_k_body"],
        )
    _refresh_representatives(profile)


def expire_profiles(profiles, current_sec):
    active_profiles = []
    expired_profiles = []
    for profile in profiles:
        expiry_at_sec = float(profile.get("expiry_at_sec") or current_sec)
        if current_sec <= expiry_at_sec:
            active_profiles.append(profile)
        else:
            expired_profiles.append(profile)
    return active_profiles, expired_profiles
