from collections import defaultdict

from .config_loader import DEFAULT_ASSOCIATION_POLICY, deep_merge


def _merge_policy(policy):
    return deep_merge(DEFAULT_ASSOCIATION_POLICY["topology_filter"], policy or {})


def build_topology_index(config):
    fps = float(config["assumed_video_fps"])
    topology = defaultdict(dict)
    for src_camera, targets in config["camera_topology"].items():
        for dst_camera, info in targets.items():
            min_sec = float(info.get("min_frame_gap", 0.0)) / fps
            max_sec = float(info.get("max_frame_gap", 0.0)) / fps
            topology[src_camera][dst_camera] = {
                "src_camera_id": src_camera,
                "dst_camera_id": dst_camera,
                "camera_id": dst_camera,
                "relation_type": info.get("relation", "weak_link"),
                "min_travel_time": min_sec,
                "avg_travel_time": (min_sec + max_sec) / 2.0,
                "max_travel_time": max_sec,
                "same_area_overlap": info.get("same_area_overlap", info.get("relation", "weak_link") == "overlap"),
                "allowed_exit_zones": list(info.get("allowed_exit_zones", [])),
                "allowed_entry_zones": list(info.get("allowed_entry_zones", [])),
            }
    return topology


def _profile_reference_points(profile):
    merged = {}
    for ref in profile.get("face_refs", []) + profile.get("body_refs", []):
        key = (ref.get("event_id", ""), ref.get("camera_id", ""), ref.get("relative_sec", ""))
        merged[key] = {
            "event_id": ref.get("event_id", ""),
            "camera_id": ref.get("camera_id", ""),
            "relative_sec": float(ref.get("relative_sec") or 0.0),
        }
    if not merged:
        key = ("", profile.get("latest_seen_camera", ""), profile.get("latest_seen_time", 0.0))
        merged[key] = {
            "event_id": "",
            "camera_id": profile.get("latest_seen_camera", ""),
            "relative_sec": float(profile.get("latest_seen_time") or 0.0),
        }
    return list(merged.values())


def _evaluate_zone_compatibility(event, profile, relation, policy):
    current_zone = event.get("zone_id", "") or ""
    profile_zones = set(profile.get("zones_seen", []) or [])
    allowed_exit = set(relation.get("allowed_exit_zones", []) or [])
    allowed_entry = set(relation.get("allowed_entry_zones", []) or [])

    if not current_zone and not profile_zones and not allowed_exit and not allowed_entry and policy["zone"]["default_allow_when_missing"]:
        return {
            "zone_allowed": True,
            "zone_score": 1.0,
            "zone_reason": "zone_unavailable_accept",
        }
    if allowed_entry and current_zone and current_zone not in allowed_entry:
        return {
            "zone_allowed": False,
            "zone_score": 0.0,
            "zone_reason": "zone_entry_reject",
        }
    if allowed_exit and profile_zones and not (profile_zones & allowed_exit):
        return {
            "zone_allowed": False,
            "zone_score": 0.0,
            "zone_reason": "zone_exit_reject",
        }
    return {
        "zone_allowed": True,
        "zone_score": 1.0,
        "zone_reason": "zone_ok",
    }


def _evaluate_time_compatibility(delta_sec, relation, policy):
    priors = policy["relation_priors"]
    relation_type = relation["relation_type"]
    if relation_type == "overlap":
        max_sec = max(float(policy["overlap"]["max_time_floor_sec"]), float(relation["max_travel_time"]))
        if abs(delta_sec) > max_sec:
            return {
                "topology_allowed": False,
                "time_score": 0.0,
                "topology_score": priors["overlap"],
                "reason_code": "overlap_window_reject",
            }
        return {
            "topology_allowed": True,
            "time_score": max(0.0, 1.0 - (abs(delta_sec) / max_sec)),
            "topology_score": priors["overlap"],
            "reason_code": "topology_time_ok",
        }
    if relation_type == "sequential":
        min_sec = float(relation["min_travel_time"])
        max_sec = max(min_sec + float(policy["sequential"]["min_window_span_sec"]), float(relation["max_travel_time"]))
        if delta_sec < min_sec or delta_sec > max_sec:
            return {
                "topology_allowed": False,
                "time_score": 0.0,
                "topology_score": priors["sequential"],
                "reason_code": "sequential_window_reject",
            }
        span = max(float(policy["sequential"]["min_window_span_sec"]), max_sec - min_sec)
        center = float(relation["avg_travel_time"])
        return {
            "topology_allowed": True,
            "time_score": max(0.0, 1.0 - (abs(delta_sec - center) / span)),
            "topology_score": priors["sequential"],
            "reason_code": "topology_time_ok",
        }
    max_sec = max(float(policy["weak_link"]["fallback_max_travel_time_sec"]), float(relation.get("max_travel_time", 2.0)))
    if (policy["weak_link"]["require_non_negative_delta"] and delta_sec < 0.0) or delta_sec > max_sec:
        return {
            "topology_allowed": False,
            "time_score": 0.0,
            "topology_score": priors["weak_link"],
            "reason_code": "weak_link_window_reject",
        }
    return {
        "topology_allowed": True,
        "time_score": max(0.0, 1.0 - (delta_sec / max_sec)),
        "topology_score": priors["weak_link"],
        "reason_code": "topology_time_ok",
    }


def evaluate_profile_topology(item, profile, topology, policy=None):
    cfg = _merge_policy(policy)
    event = item["event"]
    current_camera = event["camera_id"]
    current_sec = float(event["relative_sec"])
    if current_camera in (profile.get("history_cameras") or []):
        return {
            "candidate_unknown_global_id": profile["unknown_global_id"],
            "candidate_latest_camera": profile.get("latest_seen_camera", ""),
            "candidate_latest_time": profile.get("latest_seen_time", ""),
            "profile_camera": "",
            "profile_time": "",
            "relation_type": "camera_already_seen",
            "same_area_overlap": False,
            "min_travel_time": "",
            "avg_travel_time": "",
            "max_travel_time": "",
            "delta_sec": "",
            "topology_allowed": False,
            "zone_allowed": False,
            "time_score": 0.0,
            "topology_score": 0.0,
            "zone_score": 0.0,
            "candidate_reason": "camera_already_seen_in_profile",
        }

    best = None
    for ref in _profile_reference_points(profile):
        prev_camera = ref["camera_id"]
        prev_sec = float(ref["relative_sec"])
        relation = topology.get(prev_camera, {}).get(current_camera)
        if relation is None:
            continue
        delta_sec = current_sec - prev_sec
        time_eval = _evaluate_time_compatibility(delta_sec, relation, cfg)
        zone_eval = _evaluate_zone_compatibility(event, profile, relation, cfg)
        allowed = time_eval["topology_allowed"] and zone_eval["zone_allowed"]
        candidate = {
            "candidate_unknown_global_id": profile["unknown_global_id"],
            "candidate_latest_camera": profile.get("latest_seen_camera", ""),
            "candidate_latest_time": profile.get("latest_seen_time", ""),
            "profile_camera": prev_camera,
            "profile_time": prev_sec,
            "relation_type": relation["relation_type"],
            "same_area_overlap": relation["same_area_overlap"],
            "min_travel_time": relation["min_travel_time"],
            "avg_travel_time": relation["avg_travel_time"],
            "max_travel_time": relation["max_travel_time"],
            "delta_sec": round(delta_sec, 3),
            "topology_allowed": allowed,
            "zone_allowed": zone_eval["zone_allowed"],
            "time_score": round(float(time_eval["time_score"]), 4),
            "topology_score": round(float(time_eval["topology_score"]), 4),
            "zone_score": round(float(zone_eval["zone_score"]), 4),
            "candidate_reason": zone_eval["zone_reason"] if not zone_eval["zone_allowed"] else time_eval["reason_code"],
        }
        if best is None:
            best = candidate
            continue
        candidate_key = (
            1 if candidate["topology_allowed"] else 0,
            candidate["time_score"],
            candidate["topology_score"],
            -abs(candidate["delta_sec"]) if candidate["delta_sec"] != "" else -999999.0,
        )
        best_key = (
            1 if best["topology_allowed"] else 0,
            best["time_score"],
            best["topology_score"],
            -abs(best["delta_sec"]) if best["delta_sec"] != "" else -999999.0,
        )
        if candidate_key > best_key:
            best = candidate

    if best is None:
        return {
            "candidate_unknown_global_id": profile["unknown_global_id"],
            "candidate_latest_camera": profile.get("latest_seen_camera", ""),
            "candidate_latest_time": profile.get("latest_seen_time", ""),
            "profile_camera": "",
            "profile_time": "",
            "relation_type": "no_link",
            "same_area_overlap": False,
            "min_travel_time": "",
            "avg_travel_time": "",
            "max_travel_time": "",
            "delta_sec": "",
            "topology_allowed": False,
            "zone_allowed": False,
            "time_score": 0.0,
            "topology_score": 0.0,
            "zone_score": 0.0,
            "candidate_reason": "no_topology_path",
        }
    return best
