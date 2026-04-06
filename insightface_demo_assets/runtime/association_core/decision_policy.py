import json
from copy import deepcopy

from .appearance_evidence import cosine_similarity, evaluate_appearance_evidence
from .gallery_lifecycle import create_unknown_profile, expire_profiles, update_unknown_profile
from .quality_gate import evaluate_quality_gate
from .topology_filter import evaluate_profile_topology

DEFAULT_ASSOCIATION_POLICY = {
    "quality_gate": {},
    "gallery": {
        "top_k_face_refs": 3,
        "top_k_body_refs": 5,
        "ttl_sec": 12.0,
    },
    "known": {
        "accept_threshold": 0.65,
        "margin_threshold": 0.02,
    },
    "unknown": {
        "margin_by_relation": {
            "overlap": 0.015,
            "sequential": 0.02,
            "weak_link": 0.04,
            "camera_already_seen": 1.0,
            "no_link": 1.0,
        },
        "thresholds": {
            "overlap": {
                "face_primary": 0.18,
                "face_secondary": 0.30,
                "body_primary": 0.60,
                "body_secondary": 0.18,
            },
            "sequential": {
                "face_primary": 0.20,
                "face_secondary": 0.35,
                "body_primary": 0.58,
                "body_secondary": 0.18,
            },
            "weak_link": {
                "face_primary": 0.24,
                "face_secondary": 0.40,
                "body_primary": 0.64,
                "body_secondary": 0.22,
            },
        },
        "defer_quality_reliability_max": 0.35,
    },
}


def _deep_update(base, updates):
    merged = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sorted_known_candidates(face_embedding, identity_means):
    rows = []
    if face_embedding is None:
        return rows
    for identity_id, ref_vec in identity_means.items():
        rows.append({"identity_id": identity_id, "score": cosine_similarity(face_embedding, ref_vec)})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def best_known_match(face_embedding, identity_means):
    candidates = _sorted_known_candidates(face_embedding, identity_means)
    return candidates[0] if candidates else None


def _known_acceptance(item, identity_means, policy):
    candidates = _sorted_known_candidates(item.get("face_embedding"), identity_means)
    if not candidates:
        return None, {"reason_code": "no_known_face_embedding", "candidates": []}
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else {"identity_id": "", "score": 0.0}
    margin = best["score"] - second["score"]
    if best["score"] >= float(policy["accept_threshold"]) and margin >= float(policy["margin_threshold"]):
        return (
            {
                "identity_id": best["identity_id"],
                "score": best["score"],
                "margin": margin,
                "reason_code": "known_accept",
            },
            {"reason_code": "known_accept", "candidates": candidates[:5]},
        )
    return None, {
        "reason_code": "known_reject_open_set_fallback",
        "candidates": candidates[:5],
        "best_score": best["score"],
        "second_score": second["score"],
        "margin": margin,
    }


def _relation_thresholds(policy, relation_type):
    thresholds = policy["thresholds"]
    if relation_type in thresholds:
        return thresholds[relation_type]
    return thresholds["weak_link"]


def _candidate_acceptance(candidate, policy):
    if not candidate["quality_gate_pass"]:
        return False, "poor_quality"
    if not candidate["topology_allowed"]:
        return False, candidate["candidate_reason"] or "topology_or_time_reject"
    if not candidate["zone_allowed"]:
        return False, candidate["candidate_reason"] or "zone_reject"
    if candidate["appearance_primary"] <= 0.0:
        return False, "appearance_missing"

    thresholds = _relation_thresholds(policy, candidate["relation_type"])
    if candidate["primary_modality"] == "face":
        primary_threshold = float(thresholds["face_primary"])
        secondary_threshold = float(thresholds["face_secondary"])
    else:
        primary_threshold = float(thresholds["body_primary"])
        secondary_threshold = float(thresholds["body_secondary"])

    if candidate["appearance_primary"] < primary_threshold:
        return False, "below_primary_threshold"
    if candidate["appearance_secondary_available"] and candidate["appearance_secondary"] < secondary_threshold:
        return False, "below_secondary_threshold"
    return True, "candidate_threshold_pass"


def _ranking_key(candidate):
    return (
        1 if candidate["quality_gate_pass"] else 0,
        1 if candidate["topology_allowed"] else 0,
        candidate["appearance_primary"],
        candidate["appearance_secondary"],
        candidate["time_score"],
        candidate["quality_reliability"],
    )


def _primary_margin(top1, top2):
    if not top2:
        return 1.0
    return round(float(top1["appearance_primary"]) - float(top2["appearance_primary"]), 4)


def evaluate_profile_candidate(item, profile, topology, policy=None):
    merged_policy = _deep_update(DEFAULT_ASSOCIATION_POLICY, policy or {})
    quality = evaluate_quality_gate(item, merged_policy["quality_gate"])
    topology_eval = evaluate_profile_topology(item, profile, topology)
    appearance = evaluate_appearance_evidence(item, profile, quality)
    candidate = {
        **topology_eval,
        **appearance,
        "quality_gate_pass": quality["gate_pass"],
        "quality_reason_code": quality["reason_code"],
        "quality_reliability": quality["quality_reliability"],
        "face_quality": quality["face_quality"],
        "body_quality": quality["body_quality"],
        "face_available": appearance["face_available"],
        "body_available": appearance["body_available"],
        "appearance_secondary_available": appearance["appearance_secondary"] > 0.0,
        "final_total_score": appearance["appearance_primary"],
        "reason_code": topology_eval["candidate_reason"] if not topology_eval["topology_allowed"] else appearance["evidence_reason"],
    }
    return candidate


def _build_resolved_row(event, item, known_match=None, unknown_global_id="", decision_type="", reason_code="", resolution_source=""):
    if decision_type == "attach_known":
        identity_status = "known"
        matched_known_id = known_match["identity_id"]
        resolved_global_id = known_match["identity_id"]
    elif decision_type == "defer":
        identity_status = "deferred"
        matched_known_id = ""
        resolved_global_id = ""
    else:
        identity_status = "unknown"
        matched_known_id = ""
        resolved_global_id = unknown_global_id
    return {
        "run_mode": "mode_b_true_assoc",
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "camera_id": event["camera_id"],
        "frame_id": event["frame_id"],
        "relative_sec": event["relative_sec"],
        "global_gt_id": event["global_gt_id"],
        "anchor_camera_id": event.get("anchor_camera_id", ""),
        "anchor_relative_sec": event.get("anchor_relative_sec", ""),
        "relation_type": event.get("relation_type", ""),
        "best_head_crop": event["best_head_crop"],
        "best_body_crop": event["best_body_crop"],
        "identity_status": identity_status,
        "matched_known_id": matched_known_id,
        "matched_known_score": round(known_match["score"], 4) if known_match else "",
        "unknown_global_id": unknown_global_id,
        "resolved_global_id": resolved_global_id,
        "resolution_source": resolution_source,
        "decision_reason": reason_code,
        "reason_code": reason_code,
        "face_embedding_status": item["face_status"],
        "face_count": item["face_count"],
        "face_det_score": round(item["face_det_score"], 4) if item["face_det_score"] else "",
        "used_face_crop": item["used_face_crop"],
        "used_face_crop_path": item["used_face_crop_path"],
        "face_bbox": item["face_bbox"],
        "body_feature_status": item["body_status"],
        "body_feature_shape": item["body_shape"],
    }


def _build_trace_row(event, candidate, decision_reason, reused_old_id=False, created_new_id=False, selected_candidate=False):
    return {
        "run_mode": "mode_b_true_assoc",
        "event_id": event["event_id"],
        "event_camera_id": event["camera_id"],
        "event_relative_sec": event["relative_sec"],
        **candidate,
        "reused_old_id": reused_old_id,
        "created_new_id": created_new_id,
        "selected_candidate": selected_candidate,
        "decision_reason": decision_reason,
    }


def assign_model_identities(analyzed_events, identity_means, topology, unknown_prefix, unknown_start, policy=None):
    merged_policy = _deep_update(DEFAULT_ASSOCIATION_POLICY, policy or {})
    profiles = []
    resolved_rows = []
    trace_rows = []
    next_unknown = unknown_start

    for item in analyzed_events:
        event = item["event"]
        current_sec = float(event["relative_sec"])
        profiles, _expired_profiles = expire_profiles(profiles, current_sec)

        known_match, known_audit = _known_acceptance(item, identity_means, merged_policy["known"])
        if known_match is not None:
            resolved_rows.append(
                _build_resolved_row(
                    event,
                    item,
                    known_match=known_match,
                    decision_type="attach_known",
                    reason_code="known_accept",
                    resolution_source="model_face_match",
                )
            )
            trace_rows.append(
                {
                    "run_mode": "mode_b_true_assoc",
                    "event_id": event["event_id"],
                    "event_camera_id": event["camera_id"],
                    "event_relative_sec": event["relative_sec"],
                    "candidate_unknown_global_id": "",
                    "candidate_latest_camera": "",
                    "candidate_latest_time": "",
                    "profile_camera": "",
                    "profile_time": "",
                    "relation_type": "",
                    "same_area_overlap": "",
                    "min_travel_time": "",
                    "avg_travel_time": "",
                    "max_travel_time": "",
                    "delta_sec": "",
                    "topology_allowed": "",
                    "zone_allowed": "",
                    "time_score": "",
                    "topology_score": "",
                    "zone_score": "",
                    "face_score": round(known_match["score"], 4),
                    "body_score": "",
                    "appearance_primary": round(known_match["score"], 4),
                    "appearance_secondary": "",
                    "primary_modality": "face",
                    "modality_state": "face_only",
                    "quality_gate_pass": True,
                    "quality_reason_code": known_audit["reason_code"],
                    "quality_reliability": round(item.get("face_det_score") or 0.0, 4),
                    "final_total_score": round(known_match["score"], 4),
                    "reason_code": "known_gallery_match",
                    "reused_old_id": False,
                    "created_new_id": False,
                    "selected_candidate": False,
                    "decision_reason": "known_accept",
                }
            )
            continue

        quality = evaluate_quality_gate(item, merged_policy["quality_gate"])
        candidate_scores = []
        for profile in profiles:
            candidate = evaluate_profile_candidate(item, profile, topology, merged_policy)
            allowed, acceptance_reason = _candidate_acceptance(candidate, merged_policy["unknown"])
            candidate["acceptance_pass"] = allowed
            candidate["acceptance_reason"] = acceptance_reason
            candidate["ranking_key"] = json.dumps(_ranking_key(candidate))
            candidate_scores.append(candidate)
        candidate_scores.sort(key=_ranking_key, reverse=True)

        top1 = candidate_scores[0] if candidate_scores else None
        top2 = candidate_scores[1] if len(candidate_scores) > 1 else None
        selected = None
        margin = _primary_margin(top1, top2) if top1 else 0.0

        if quality["gate_pass"] and top1 and top1["acceptance_pass"]:
            required_margin = float(
                merged_policy["unknown"]["margin_by_relation"].get(
                    top1["relation_type"],
                    merged_policy["unknown"]["margin_by_relation"]["weak_link"],
                )
            )
            if margin >= required_margin:
                selected = top1
                selected["decision_reason"] = "unknown_reuse"
                selected["reason_code"] = "unknown_reuse"
            else:
                top1["decision_reason"] = "ambiguous_margin_reject"
                top1["reason_code"] = "ambiguous"

        if selected is not None:
            profile = next(profile for profile in profiles if profile["unknown_global_id"] == selected["candidate_unknown_global_id"])
            update_unknown_profile(profile, item, policy=merged_policy["gallery"])
            resolved_rows.append(
                _build_resolved_row(
                    event,
                    item,
                    unknown_global_id=profile["unknown_global_id"],
                    decision_type="reuse_unknown",
                    reason_code="unknown_reuse",
                    resolution_source="model_unknown_gallery_reuse",
                )
            )
            for candidate in candidate_scores:
                decision_reason = candidate.get("decision_reason") or candidate.get("acceptance_reason") or candidate.get("candidate_reason")
                trace_rows.append(
                    _build_trace_row(
                        event,
                        candidate,
                        decision_reason=decision_reason,
                        reused_old_id=candidate["candidate_unknown_global_id"] == profile["unknown_global_id"],
                        created_new_id=False,
                        selected_candidate=candidate["candidate_unknown_global_id"] == profile["unknown_global_id"],
                    )
                )
            continue

        if not quality["gate_pass"]:
            resolved_rows.append(
                _build_resolved_row(
                    event,
                    item,
                    decision_type="defer",
                    reason_code=quality["reason_code"],
                    resolution_source="model_defer_quality_gate",
                )
            )
            if not candidate_scores:
                trace_rows.append(
                    {
                        "run_mode": "mode_b_true_assoc",
                        "event_id": event["event_id"],
                        "event_camera_id": event["camera_id"],
                        "event_relative_sec": event["relative_sec"],
                        "candidate_unknown_global_id": "",
                        "candidate_latest_camera": "",
                        "candidate_latest_time": "",
                        "profile_camera": "",
                        "profile_time": "",
                        "relation_type": "",
                        "same_area_overlap": "",
                        "min_travel_time": "",
                        "avg_travel_time": "",
                        "max_travel_time": "",
                        "delta_sec": "",
                        "topology_allowed": "",
                        "zone_allowed": "",
                        "time_score": "",
                        "topology_score": "",
                        "zone_score": "",
                        "face_score": "",
                        "body_score": "",
                        "appearance_primary": "",
                        "appearance_secondary": "",
                        "primary_modality": quality["primary_modality"],
                        "modality_state": quality["modality_state"],
                        "quality_gate_pass": False,
                        "quality_reason_code": quality["reason_code"],
                        "quality_reliability": quality["quality_reliability"],
                        "final_total_score": "",
                        "reason_code": quality["reason_code"],
                        "reused_old_id": False,
                        "created_new_id": False,
                        "selected_candidate": False,
                        "decision_reason": quality["reason_code"],
                    }
                )
            for candidate in candidate_scores:
                trace_rows.append(
                    _build_trace_row(
                        event,
                        candidate,
                        decision_reason=quality["reason_code"],
                        reused_old_id=False,
                        created_new_id=False,
                        selected_candidate=False,
                    )
                )
            continue

        unknown_global_id = f"{unknown_prefix}_{next_unknown:04d}"
        next_unknown += 1
        profile = create_unknown_profile(unknown_global_id, item, policy=merged_policy["gallery"])
        had_previous_profiles = bool(profiles)
        profiles.append(profile)
        create_reason = "no_candidate"
        if top1 is not None:
            if top1.get("reason_code") == "ambiguous":
                if quality["quality_reliability"] <= float(merged_policy["unknown"]["defer_quality_reliability_max"]):
                    create_reason = "ambiguous_low_quality_create"
                else:
                    create_reason = "ambiguous_create_unknown"
            else:
                create_reason = top1.get("acceptance_reason") or top1.get("candidate_reason") or "below_threshold"
        elif not had_previous_profiles:
            create_reason = "no_previous_unknown_profiles"
        resolved_rows.append(
            _build_resolved_row(
                event,
                item,
                unknown_global_id=unknown_global_id,
                decision_type="create_unknown",
                reason_code=create_reason,
                resolution_source="model_unknown_new_profile",
            )
        )
        if not candidate_scores:
            trace_rows.append(
                {
                    "run_mode": "mode_b_true_assoc",
                    "event_id": event["event_id"],
                    "event_camera_id": event["camera_id"],
                    "event_relative_sec": event["relative_sec"],
                    "candidate_unknown_global_id": "",
                    "candidate_latest_camera": "",
                    "candidate_latest_time": "",
                    "profile_camera": "",
                    "profile_time": "",
                    "relation_type": "",
                    "same_area_overlap": "",
                    "min_travel_time": "",
                    "avg_travel_time": "",
                    "max_travel_time": "",
                    "delta_sec": "",
                    "topology_allowed": "",
                    "zone_allowed": "",
                    "time_score": "",
                    "topology_score": "",
                    "zone_score": "",
                    "face_score": "",
                    "body_score": "",
                    "appearance_primary": "",
                    "appearance_secondary": "",
                    "primary_modality": quality["primary_modality"],
                    "modality_state": quality["modality_state"],
                    "quality_gate_pass": quality["gate_pass"],
                    "quality_reason_code": quality["reason_code"],
                    "quality_reliability": quality["quality_reliability"],
                    "final_total_score": "",
                    "reason_code": create_reason,
                    "reused_old_id": False,
                    "created_new_id": True,
                    "selected_candidate": False,
                    "decision_reason": create_reason,
                }
            )
        for candidate in candidate_scores:
            decision_reason = candidate.get("decision_reason") or candidate.get("acceptance_reason") or create_reason
            trace_rows.append(
                _build_trace_row(
                    event,
                    candidate,
                    decision_reason=decision_reason,
                    reused_old_id=False,
                    created_new_id=True,
                    selected_candidate=False,
                )
            )
    return resolved_rows, profiles, trace_rows
