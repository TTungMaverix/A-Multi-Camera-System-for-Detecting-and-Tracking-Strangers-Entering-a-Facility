from collections import defaultdict

from .config_loader import DEFAULT_ASSOCIATION_POLICY, deep_merge


def _merge_policy(policy):
    return deep_merge(DEFAULT_ASSOCIATION_POLICY["topology_filter"], policy or {})


def _legacy_transition_dict(config):
    fps = float(config["assumed_video_fps"])
    transitions = []
    for src_camera, targets in config["camera_topology"].items():
        for dst_camera, info in targets.items():
            min_sec = float(info.get("min_frame_gap", 0.0)) / fps
            max_sec = float(info.get("max_frame_gap", 0.0)) / fps
            transitions.append(
                {
                    "transition_id": f"{src_camera}_to_{dst_camera}_{info.get('relation', 'weak_link')}",
                    "src_camera_id": src_camera,
                    "dst_camera_id": dst_camera,
                    "relation_type": info.get("relation", "weak_link"),
                    "allowed_relation_types": [info.get("relation", "weak_link")],
                    "same_area_overlap": info.get("same_area_overlap", info.get("relation", "weak_link") == "overlap"),
                    "min_travel_time_sec": min_sec,
                    "avg_travel_time_sec": (min_sec + max_sec) / 2.0,
                    "max_travel_time_sec": max_sec,
                    "allowed_exit_zones": list(info.get("allowed_exit_zones", [])),
                    "allowed_entry_zones": list(info.get("allowed_entry_zones", [])),
                }
            )
    return transitions


def build_topology_index(config):
    topology = defaultdict(dict)
    if "transitions" in config:
        transitions = config.get("transitions", [])
    else:
        transitions = _legacy_transition_dict(config)

    for info in transitions:
        src_camera = info["src_camera_id"]
        dst_camera = info["dst_camera_id"]
        relation_type = info.get("relation_type", "weak_link")
        min_sec = float(info.get("min_travel_time_sec", info.get("min_travel_time", 0.0)) or 0.0)
        avg_sec = float(info.get("avg_travel_time_sec", info.get("avg_travel_time", min_sec)) or min_sec)
        max_sec = float(info.get("max_travel_time_sec", info.get("max_travel_time", avg_sec)) or avg_sec)
        topology[src_camera][dst_camera] = {
            "transition_id": info.get("transition_id", f"{src_camera}_to_{dst_camera}_{relation_type}"),
            "src_camera_id": src_camera,
            "dst_camera_id": dst_camera,
            "camera_id": dst_camera,
            "relation_type": relation_type,
            "allowed_relation_types": list(info.get("allowed_relation_types", [relation_type])),
            "min_travel_time": min_sec,
            "avg_travel_time": avg_sec,
            "max_travel_time": max_sec,
            "same_area_overlap": bool(info.get("same_area_overlap", relation_type == "overlap")),
            "allowed_exit_zones": list(info.get("allowed_exit_zones", [])),
            "allowed_entry_zones": list(info.get("allowed_entry_zones", [])),
            "allowed_exit_subzones": list(info.get("allowed_exit_subzones", [])),
            "allowed_entry_subzones": list(info.get("allowed_entry_subzones", [])),
            "weak_link_support": bool(info.get("weak_link_support", relation_type == "weak_link")),
            "description": info.get("description", ""),
        }
    return topology


def _profile_reference_points(profile):
    merged = {}
    for ref in profile.get("face_refs", []) + profile.get("body_refs", []):
        key = (
            ref.get("event_id", ""),
            ref.get("camera_id", ""),
            ref.get("relative_sec", ""),
            ref.get("zone_id", ""),
            ref.get("subzone_id", ""),
        )
        merged[key] = {
            "event_id": ref.get("event_id", ""),
            "camera_id": ref.get("camera_id", ""),
            "relative_sec": float(ref.get("relative_sec") or 0.0),
            "zone_id": ref.get("zone_id", ""),
            "zone_type": ref.get("zone_type", ""),
            "subzone_id": ref.get("subzone_id", ""),
            "subzone_type": ref.get("subzone_type", ""),
        }
    if not merged:
        key = (
            "",
            profile.get("latest_seen_camera", ""),
            profile.get("latest_seen_time", 0.0),
            profile.get("latest_seen_zone", ""),
            profile.get("latest_seen_subzone", ""),
        )
        merged[key] = {
            "event_id": "",
            "camera_id": profile.get("latest_seen_camera", ""),
            "relative_sec": float(profile.get("latest_seen_time") or 0.0),
            "zone_id": profile.get("latest_seen_zone", ""),
            "zone_type": profile.get("latest_seen_zone_type", ""),
            "subzone_id": profile.get("latest_seen_subzone", ""),
            "subzone_type": profile.get("latest_seen_subzone_type", ""),
        }
    return list(merged.values())


def _evaluate_zone_compatibility(event, profile, relation, ref, policy):
    zone_cfg = policy["zone"]
    target_zone = event.get("zone_id", "") or ""
    source_zone = ref.get("zone_id", "") or profile.get("latest_seen_zone", "") or ""
    profile_zones = {zone for zone in (profile.get("zones_seen", []) or []) if zone}
    if source_zone:
        profile_zones.add(source_zone)
    allowed_exit = {zone for zone in relation.get("allowed_exit_zones", []) or [] if zone}
    allowed_entry = {zone for zone in relation.get("allowed_entry_zones", []) or [] if zone}

    if not allowed_exit and not allowed_entry:
        return {
            "zone_valid": True,
            "zone_score": 1.0,
            "zone_reason": "zone_not_required",
            "source_zone_id": source_zone,
            "target_zone_id": target_zone,
            "fallback_without_zone": False,
        }

    fallback_without_zone = False
    zone_reason = "zone_ok"

    if allowed_exit:
        if source_zone:
            if source_zone not in allowed_exit:
                return {
                    "zone_valid": False,
                    "zone_score": 0.0,
                    "zone_reason": "zone_exit_reject",
                    "source_zone_id": source_zone,
                    "target_zone_id": target_zone,
                    "fallback_without_zone": False,
                }
        elif profile_zones:
            if not (profile_zones & allowed_exit):
                return {
                    "zone_valid": False,
                    "zone_score": 0.0,
                    "zone_reason": "zone_exit_reject",
                    "source_zone_id": "",
                    "target_zone_id": target_zone,
                    "fallback_without_zone": False,
                }
        elif zone_cfg["default_allow_when_missing"]:
            fallback_without_zone = True
            zone_reason = "zone_exit_missing_fallback"
        else:
            return {
                "zone_valid": False,
                "zone_score": 0.0,
                "zone_reason": "zone_exit_missing_reject",
                "source_zone_id": "",
                "target_zone_id": target_zone,
                "fallback_without_zone": False,
            }

    if allowed_entry:
        if target_zone:
            if target_zone not in allowed_entry:
                return {
                    "zone_valid": False,
                    "zone_score": 0.0,
                    "zone_reason": "zone_entry_reject",
                    "source_zone_id": source_zone,
                    "target_zone_id": target_zone,
                    "fallback_without_zone": False,
                }
        elif zone_cfg["default_allow_when_missing"]:
            fallback_without_zone = True
            if zone_reason == "zone_ok":
                zone_reason = "zone_entry_missing_fallback"
        else:
            return {
                "zone_valid": False,
                "zone_score": 0.0,
                "zone_reason": "zone_entry_missing_reject",
                "source_zone_id": source_zone,
                "target_zone_id": "",
                "fallback_without_zone": False,
            }

    return {
        "zone_valid": True,
        "zone_score": 1.0,
        "zone_reason": zone_reason,
        "source_zone_id": source_zone,
        "target_zone_id": target_zone,
        "fallback_without_zone": fallback_without_zone,
    }


def _evaluate_subzone_compatibility(event, profile, relation, ref, policy):
    subzone_cfg = policy["subzone"]
    target_subzone = event.get("subzone_id", "") or ""
    source_subzone = ref.get("subzone_id", "") or profile.get("latest_seen_subzone", "") or ""
    profile_subzones = {subzone for subzone in (profile.get("subzones_seen", []) or []) if subzone}
    if source_subzone:
        profile_subzones.add(source_subzone)
    allowed_exit = {subzone for subzone in relation.get("allowed_exit_subzones", []) or [] if subzone}
    allowed_entry = {subzone for subzone in relation.get("allowed_entry_subzones", []) or [] if subzone}

    if not allowed_exit and not allowed_entry:
        return {
            "subzone_valid": True,
            "subzone_score": 1.0,
            "subzone_reason": "subzone_not_required",
            "source_subzone_id": source_subzone,
            "target_subzone_id": target_subzone,
            "fallback_without_subzone": False,
        }

    fallback_without_subzone = False
    subzone_reason = "subzone_ok"

    if allowed_exit:
        if source_subzone:
            if source_subzone not in allowed_exit:
                return {
                    "subzone_valid": False,
                    "subzone_score": 0.0,
                    "subzone_reason": "subzone_exit_reject",
                    "source_subzone_id": source_subzone,
                    "target_subzone_id": target_subzone,
                    "fallback_without_subzone": False,
                }
        elif profile_subzones:
            if not (profile_subzones & allowed_exit):
                return {
                    "subzone_valid": False,
                    "subzone_score": 0.0,
                    "subzone_reason": "subzone_exit_reject",
                    "source_subzone_id": "",
                    "target_subzone_id": target_subzone,
                    "fallback_without_subzone": False,
                }
        elif subzone_cfg["default_allow_when_missing"]:
            fallback_without_subzone = True
            subzone_reason = "subzone_exit_missing_fallback"
        else:
            return {
                "subzone_valid": False,
                "subzone_score": 0.0,
                "subzone_reason": "subzone_exit_missing_reject",
                "source_subzone_id": "",
                "target_subzone_id": target_subzone,
                "fallback_without_subzone": False,
            }

    if allowed_entry:
        if target_subzone:
            if target_subzone not in allowed_entry:
                return {
                    "subzone_valid": False,
                    "subzone_score": 0.0,
                    "subzone_reason": "subzone_entry_reject",
                    "source_subzone_id": source_subzone,
                    "target_subzone_id": target_subzone,
                    "fallback_without_subzone": False,
                }
        elif subzone_cfg["default_allow_when_missing"]:
            fallback_without_subzone = True
            if subzone_reason == "subzone_ok":
                subzone_reason = "subzone_entry_missing_fallback"
        else:
            return {
                "subzone_valid": False,
                "subzone_score": 0.0,
                "subzone_reason": "subzone_entry_missing_reject",
                "source_subzone_id": source_subzone,
                "target_subzone_id": "",
                "fallback_without_subzone": False,
            }

    return {
        "subzone_valid": True,
        "subzone_score": 1.0,
        "subzone_reason": subzone_reason,
        "source_subzone_id": source_subzone,
        "target_subzone_id": target_subzone,
        "fallback_without_subzone": fallback_without_subzone,
    }


def _evaluate_time_compatibility(delta_sec, relation, policy):
    priors = policy["relation_priors"]
    relation_type = relation["relation_type"]
    if relation_type == "overlap":
        max_sec = max(float(policy["overlap"]["max_time_floor_sec"]), float(relation["max_travel_time"]))
        if abs(delta_sec) > max_sec:
            return {
                "time_valid": False,
                "time_score": 0.0,
                "topology_score": priors["overlap"],
                "time_reason": "overlap_window_reject",
            }
        return {
            "time_valid": True,
            "time_score": max(0.0, 1.0 - (abs(delta_sec) / max_sec)),
            "topology_score": priors["overlap"],
            "time_reason": "topology_time_ok",
        }
    if relation_type == "sequential":
        min_sec = float(relation["min_travel_time"])
        max_sec = max(min_sec + float(policy["sequential"]["min_window_span_sec"]), float(relation["max_travel_time"]))
        if delta_sec < min_sec or delta_sec > max_sec:
            return {
                "time_valid": False,
                "time_score": 0.0,
                "topology_score": priors["sequential"],
                "time_reason": "sequential_window_reject",
            }
        span = max(float(policy["sequential"]["min_window_span_sec"]), max_sec - min_sec)
        center = float(relation["avg_travel_time"])
        return {
            "time_valid": True,
            "time_score": max(0.0, 1.0 - (abs(delta_sec - center) / span)),
            "topology_score": priors["sequential"],
            "time_reason": "topology_time_ok",
        }
    max_sec = max(float(policy["weak_link"]["fallback_max_travel_time_sec"]), float(relation.get("max_travel_time", 2.0)))
    if (policy["weak_link"]["require_non_negative_delta"] and delta_sec < 0.0) or delta_sec > max_sec:
        return {
            "time_valid": False,
            "time_score": 0.0,
            "topology_score": priors["weak_link"],
            "time_reason": "weak_link_window_reject",
        }
    return {
        "time_valid": True,
        "time_score": max(0.0, 1.0 - (delta_sec / max_sec)),
        "topology_score": priors["weak_link"],
        "time_reason": "topology_time_ok",
    }


def _blocked_candidate(profile, reason_code):
    return {
        "candidate_unknown_global_id": profile["unknown_global_id"],
        "candidate_latest_camera": profile.get("latest_seen_camera", ""),
        "candidate_latest_time": profile.get("latest_seen_time", ""),
        "profile_camera": "",
        "profile_time": "",
        "source_zone_id": "",
        "target_zone_id": "",
        "source_subzone_id": "",
        "target_subzone_id": "",
        "relation_type": "camera_already_seen" if reason_code == "camera_already_seen_in_profile" else "no_link",
        "same_area_overlap": False,
        "transition_rule_used": "",
        "min_travel_time": "",
        "avg_travel_time": "",
        "max_travel_time": "",
        "travel_window": {},
        "delta_sec": "",
        "topology_valid": False,
        "time_valid": False,
        "zone_valid": False,
        "subzone_valid": False,
        "topology_allowed": False,
        "zone_allowed": False,
        "time_score": 0.0,
        "topology_score": 0.0,
        "zone_score": 0.0,
        "subzone_score": 0.0,
        "zone_reason": reason_code,
        "subzone_reason": reason_code,
        "time_reason": reason_code,
        "fallback_without_zone": False,
        "fallback_without_subzone": False,
        "candidate_reason": reason_code,
        "rejection_reason": reason_code,
    }


def evaluate_profile_topology(item, profile, topology, policy=None):
    cfg = _merge_policy(policy)
    event = item["event"]
    current_camera = event["camera_id"]
    current_sec = float(event["relative_sec"])
    if current_camera in (profile.get("history_cameras") or []):
        return _blocked_candidate(profile, "camera_already_seen_in_profile")

    best = None
    for ref in _profile_reference_points(profile):
        prev_camera = ref["camera_id"]
        prev_sec = float(ref["relative_sec"])
        relation = topology.get(prev_camera, {}).get(current_camera)
        if relation is None:
            continue
        delta_sec = current_sec - prev_sec
        time_eval = _evaluate_time_compatibility(delta_sec, relation, cfg)
        zone_eval = _evaluate_zone_compatibility(event, profile, relation, ref, cfg)
        subzone_eval = _evaluate_subzone_compatibility(event, profile, relation, ref, cfg)
        time_valid = time_eval["time_valid"]
        zone_valid = zone_eval["zone_valid"]
        subzone_valid = subzone_eval["subzone_valid"]
        allowed = time_valid and zone_valid and subzone_valid
        rejection_reason = ""
        if not time_valid:
            rejection_reason = time_eval["time_reason"]
        elif not subzone_valid:
            rejection_reason = subzone_eval["subzone_reason"]
        elif not zone_valid:
            rejection_reason = zone_eval["zone_reason"]
        candidate_reason = (
            subzone_eval["subzone_reason"]
            if subzone_eval["fallback_without_subzone"]
            else (
                zone_eval["zone_reason"]
                if zone_eval["fallback_without_zone"]
                else (rejection_reason or time_eval["time_reason"])
            )
        )
        candidate = {
            "candidate_unknown_global_id": profile["unknown_global_id"],
            "candidate_latest_camera": profile.get("latest_seen_camera", ""),
            "candidate_latest_time": profile.get("latest_seen_time", ""),
            "profile_camera": prev_camera,
            "profile_time": prev_sec,
            "source_zone_id": zone_eval["source_zone_id"],
            "target_zone_id": zone_eval["target_zone_id"],
            "source_subzone_id": subzone_eval["source_subzone_id"],
            "target_subzone_id": subzone_eval["target_subzone_id"],
            "relation_type": relation["relation_type"],
            "same_area_overlap": relation["same_area_overlap"],
            "transition_rule_used": relation.get("transition_id", f"{prev_camera}_to_{current_camera}_{relation['relation_type']}"),
            "min_travel_time": relation["min_travel_time"],
            "avg_travel_time": relation["avg_travel_time"],
            "max_travel_time": relation["max_travel_time"],
            "travel_window": {
                "min_travel_time": relation["min_travel_time"],
                "avg_travel_time": relation["avg_travel_time"],
                "max_travel_time": relation["max_travel_time"],
            },
            "delta_sec": round(delta_sec, 3),
            "topology_valid": True,
            "time_valid": time_valid,
            "zone_valid": zone_valid,
            "subzone_valid": subzone_valid,
            "topology_allowed": allowed,
            "zone_allowed": zone_valid,
            "time_score": round(float(time_eval["time_score"]), 4),
            "topology_score": round(float(time_eval["topology_score"]), 4),
            "zone_score": round(float(zone_eval["zone_score"]), 4),
            "subzone_score": round(float(subzone_eval["subzone_score"]), 4),
            "zone_reason": zone_eval["zone_reason"],
            "subzone_reason": subzone_eval["subzone_reason"],
            "time_reason": time_eval["time_reason"],
            "fallback_without_zone": zone_eval["fallback_without_zone"],
            "fallback_without_subzone": subzone_eval["fallback_without_subzone"],
            "candidate_reason": candidate_reason,
            "rejection_reason": rejection_reason,
        }
        if best is None:
            best = candidate
            continue
        candidate_key = (
            1 if candidate["topology_allowed"] else 0,
            1 if candidate["time_valid"] else 0,
            1 if candidate["zone_valid"] else 0,
            1 if candidate["subzone_valid"] else 0,
            candidate["time_score"],
            candidate["topology_score"],
            -abs(candidate["delta_sec"]) if candidate["delta_sec"] != "" else -999999.0,
        )
        best_key = (
            1 if best["topology_allowed"] else 0,
            1 if best["time_valid"] else 0,
            1 if best["zone_valid"] else 0,
            1 if best["subzone_valid"] else 0,
            best["time_score"],
            best["topology_score"],
            -abs(best["delta_sec"]) if best["delta_sec"] != "" else -999999.0,
        )
        if candidate_key > best_key:
            best = candidate

    if best is None:
        return _blocked_candidate(profile, "no_topology_path")
    return best
