import json

from .appearance_evidence import cosine_similarity, evaluate_appearance_evidence
from .config_loader import DEFAULT_ASSOCIATION_POLICY, deep_merge
from .gallery_lifecycle import create_unknown_profile, expire_profiles, update_unknown_profile
from .quality_gate import evaluate_quality_gate
from .topology_filter import evaluate_profile_topology


def _merge_policy(policy):
    return deep_merge(DEFAULT_ASSOCIATION_POLICY, policy or {})


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
        return None, {
            "reason_code": "no_known_face_embedding",
            "candidates": [],
            "thresholds_used": {
                "known_accept_threshold": float(policy["known_accept_threshold"]),
                "known_margin_threshold": float(policy["known_margin_threshold"]),
            },
        }
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else {"identity_id": "", "score": 0.0}
    margin = best["score"] - second["score"]
    thresholds = {
        "known_accept_threshold": float(policy["known_accept_threshold"]),
        "known_margin_threshold": float(policy["known_margin_threshold"]),
    }
    if best["score"] >= thresholds["known_accept_threshold"] and margin >= thresholds["known_margin_threshold"]:
        return (
            {
                "identity_id": best["identity_id"],
                "score": best["score"],
                "margin": margin,
                "reason_code": "known_accept",
            },
            {"reason_code": "known_accept", "candidates": candidates[:5], "thresholds_used": thresholds},
        )
    return None, {
        "reason_code": "known_reject_open_set_fallback",
        "candidates": candidates[:5],
        "best_score": best["score"],
        "second_score": second["score"],
        "margin": margin,
        "thresholds_used": thresholds,
    }


def _relation_thresholds(policy, relation_type):
    thresholds = policy["relation_thresholds"]
    if relation_type in thresholds:
        return thresholds[relation_type]
    return thresholds["weak_link"]


def _candidate_acceptance(candidate, policy):
    if not candidate["quality_gate_pass"]:
        return False, "poor_quality", {}
    if not candidate.get("topology_valid", candidate["topology_allowed"]):
        return False, candidate.get("rejection_reason") or candidate["candidate_reason"] or "topology_reject", {}
    if not candidate.get("time_valid", candidate["topology_allowed"]):
        return False, candidate.get("time_reason") or candidate.get("rejection_reason") or "time_reject", {}
    if not candidate.get("zone_valid", candidate["zone_allowed"]):
        return False, candidate.get("zone_reason") or candidate.get("rejection_reason") or "zone_reject", {}
    if not candidate["topology_allowed"]:
        return False, candidate.get("rejection_reason") or candidate["candidate_reason"] or "topology_or_time_reject", {}
    if candidate["appearance_primary"] <= 0.0:
        return False, "appearance_missing", {}

    thresholds = _relation_thresholds(policy, candidate["relation_type"])
    primary_floor = float(policy["unknown_reuse_threshold"])
    minimum_evidence = policy["minimum_evidence"]
    if candidate["primary_modality"] == "face":
        primary_threshold = max(primary_floor, float(thresholds["face_primary"]))
        secondary_threshold = float(thresholds["face_secondary"])
    else:
        primary_threshold = max(primary_floor, float(thresholds["body_primary"]))
        secondary_threshold = float(thresholds["body_secondary"])
    threshold_info = {
        "unknown_reuse_threshold": primary_floor,
        "relation_type": candidate["relation_type"],
        "primary_threshold": primary_threshold,
        "secondary_threshold": secondary_threshold,
        "minimum_evidence": minimum_evidence,
    }

    if candidate["appearance_primary"] < primary_threshold:
        return False, "below_primary_threshold", threshold_info
    if (
        minimum_evidence["require_secondary_when_available"]
        and candidate["appearance_secondary_available"]
        and candidate["appearance_secondary"] < secondary_threshold
    ):
        return False, "below_secondary_threshold", threshold_info
    return True, "candidate_threshold_pass", threshold_info


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
    merged_policy = _merge_policy(policy)
    quality = evaluate_quality_gate(item, merged_policy["quality_gate"])
    quality["appearance_evidence_policy"] = merged_policy["appearance_evidence"]
    topology_eval = evaluate_profile_topology(item, profile, topology, merged_policy["topology_filter"])
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
        "appearance_secondary_available": appearance["appearance_secondary"]
        > float(merged_policy["appearance_evidence"]["secondary_available_min_score"]),
        "final_total_score": appearance["appearance_primary"],
        "reason_code": (
            topology_eval.get("rejection_reason")
            or topology_eval["candidate_reason"]
            if not topology_eval["topology_allowed"]
            else appearance["evidence_reason"]
        ),
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
        "zone_id": event.get("zone_id", ""),
        "zone_type": event.get("zone_type", ""),
        "zone_reason": event.get("zone_reason", ""),
        "zone_fallback_used": event.get("zone_fallback_used", False),
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
        "event_zone_id": event.get("zone_id", ""),
        "event_zone_type": event.get("zone_type", ""),
        **candidate,
        "reused_old_id": reused_old_id,
        "created_new_id": created_new_id,
        "selected_candidate": selected_candidate,
        "decision_reason": decision_reason,
    }


def _build_decision_log(
    event,
    quality,
    candidate_scores,
    decision,
    reason_code,
    selected_candidate,
    gallery_id_before,
    gallery_id_after,
    thresholds_used,
    margin_used,
):
    candidate_set_before_filter = [candidate["candidate_unknown_global_id"] for candidate in candidate_scores if candidate["candidate_unknown_global_id"]]
    candidate_set_after_filter = [
        candidate["candidate_unknown_global_id"]
        for candidate in candidate_scores
        if candidate["topology_allowed"]
    ]
    top_candidate = selected_candidate or (candidate_scores[0] if candidate_scores else None)
    candidate_details = []
    for candidate in candidate_scores:
        candidate_details.append(
            {
                "candidate_unknown_global_id": candidate["candidate_unknown_global_id"],
                "relation_type": candidate["relation_type"],
                "transition_rule_used": candidate.get("transition_rule_used", ""),
                "source_zone_id": candidate.get("source_zone_id", ""),
                "target_zone_id": candidate.get("target_zone_id", ""),
                "topology_valid": candidate.get("topology_valid", candidate["topology_allowed"]),
                "time_valid": candidate.get("time_valid", candidate["topology_allowed"]),
                "zone_valid": candidate.get("zone_valid", candidate["zone_allowed"]),
                "zone_reason": candidate.get("zone_reason", ""),
                "time_reason": candidate.get("time_reason", ""),
                "rejection_reason": candidate.get("rejection_reason", ""),
                "fallback_without_zone": candidate.get("fallback_without_zone", False),
                "topology_allowed": candidate["topology_allowed"],
                "zone_allowed": candidate["zone_allowed"],
                "candidate_reason": candidate["candidate_reason"],
                "acceptance_reason": candidate.get("acceptance_reason", ""),
                "reason_code": candidate.get("reason_code", ""),
                "quality_gate_pass": candidate["quality_gate_pass"],
                "face_score": candidate["face_score"],
                "body_score": candidate["body_score"],
                "appearance_primary": candidate["appearance_primary"],
                "appearance_secondary": candidate["appearance_secondary"],
                "time_score": candidate["time_score"],
                "topology_score": candidate["topology_score"],
                "zone_score": candidate["zone_score"],
                "quality_reliability": candidate["quality_reliability"],
                "selected_candidate": selected_candidate is not None
                and candidate["candidate_unknown_global_id"] == selected_candidate["candidate_unknown_global_id"],
            }
        )
    return {
        "timestamp_sec": float(event["relative_sec"]),
        "relative_time": float(event["relative_sec"]),
        "camera_id": event["camera_id"],
        "observation_id": event["event_id"],
        "event_type": event["event_type"],
        "zone_id": event.get("zone_id", ""),
        "zone_type": event.get("zone_type", ""),
        "quality_gate_pass": quality["gate_pass"],
        "quality_gate_reason": quality["reason_code"],
        "candidate_set_before_filter": candidate_set_before_filter,
        "candidate_set_after_filter": candidate_set_after_filter,
        "selected_candidate_id": selected_candidate["candidate_unknown_global_id"] if selected_candidate else "",
        "relation_type": top_candidate["relation_type"] if top_candidate else "",
        "transition_rule_used": top_candidate.get("transition_rule_used", "") if top_candidate else "",
        "topology_metadata": {
            "profile_camera": top_candidate["profile_camera"] if top_candidate else "",
            "topology_valid": top_candidate.get("topology_valid", False) if top_candidate else False,
            "same_area_overlap": top_candidate["same_area_overlap"] if top_candidate else False,
            "topology_allowed": top_candidate["topology_allowed"] if top_candidate else False,
            "time_valid": top_candidate.get("time_valid", False) if top_candidate else False,
            "zone_valid": top_candidate.get("zone_valid", False) if top_candidate else False,
            "zone_allowed": top_candidate["zone_allowed"] if top_candidate else False,
        },
        "time_delta": top_candidate["delta_sec"] if top_candidate else "",
        "travel_window": top_candidate.get("travel_window", {}) if top_candidate else {},
        "source_zone_id": top_candidate.get("source_zone_id", "") if top_candidate else "",
        "target_zone_id": top_candidate.get("target_zone_id", event.get("zone_id", "")) if top_candidate else event.get("zone_id", ""),
        "zone_valid": top_candidate.get("zone_valid", False) if top_candidate else False,
        "zone_reason": top_candidate.get("zone_reason", "") if top_candidate else event.get("zone_reason", ""),
        "fallback_without_zone": top_candidate.get("fallback_without_zone", False) if top_candidate else bool(event.get("zone_fallback_used", False)),
        "modality_primary": top_candidate["primary_modality"] if top_candidate else quality["primary_modality"],
        "modality_secondary": "body"
        if top_candidate and top_candidate["primary_modality"] == "face"
        else ("face" if top_candidate and top_candidate["primary_modality"] == "body" else ""),
        "face_score": top_candidate["face_score"] if top_candidate else "",
        "body_score": top_candidate["body_score"] if top_candidate else "",
        "thresholds_used": thresholds_used,
        "margin_used": margin_used,
        "decision": decision,
        "reason_code": reason_code,
        "gallery_id_before": gallery_id_before,
        "gallery_id_after": gallery_id_after,
        "candidate_evaluations": candidate_details,
    }


def assign_model_identities(
    analyzed_events,
    identity_means,
    topology,
    unknown_prefix,
    unknown_start,
    policy=None,
    return_debug_bundle=False,
):
    merged_policy = _merge_policy(policy)
    profiles = []
    resolved_rows = []
    trace_rows = []
    decision_logs = []
    next_unknown = unknown_start
    decision_cfg = merged_policy["decision_policy"]
    gallery_cfg = merged_policy["gallery_lifecycle"]

    for item in analyzed_events:
        event = item["event"]
        current_sec = float(event["relative_sec"])
        profiles, _expired_profiles = expire_profiles(profiles, current_sec)

        known_match, known_audit = _known_acceptance(item, identity_means, decision_cfg)
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
                    "event_zone_id": event.get("zone_id", ""),
                    "event_zone_type": event.get("zone_type", ""),
                    "candidate_unknown_global_id": "",
                    "candidate_latest_camera": "",
                    "candidate_latest_time": "",
                    "profile_camera": "",
                    "profile_time": "",
                    "source_zone_id": "",
                    "target_zone_id": event.get("zone_id", ""),
                    "relation_type": "",
                    "same_area_overlap": "",
                    "transition_rule_used": "",
                    "min_travel_time": "",
                    "avg_travel_time": "",
                    "max_travel_time": "",
                    "travel_window": {},
                    "delta_sec": "",
                    "topology_valid": "",
                    "time_valid": "",
                    "zone_valid": "",
                    "topology_allowed": "",
                    "zone_allowed": "",
                    "time_score": "",
                    "topology_score": "",
                    "zone_score": "",
                    "zone_reason": event.get("zone_reason", ""),
                    "time_reason": "",
                    "rejection_reason": "",
                    "fallback_without_zone": bool(event.get("zone_fallback_used", False)),
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
            decision_logs.append(
                {
                    "timestamp_sec": float(event["relative_sec"]),
                    "relative_time": float(event["relative_sec"]),
                    "camera_id": event["camera_id"],
                    "observation_id": event["event_id"],
                    "event_type": event["event_type"],
                    "zone_id": event.get("zone_id", ""),
                    "zone_type": event.get("zone_type", ""),
                    "quality_gate_pass": True,
                    "quality_gate_reason": "quality_gate_pass",
                    "candidate_set_before_filter": [],
                    "candidate_set_after_filter": [],
                    "selected_candidate_id": "",
                    "relation_type": "",
                    "transition_rule_used": "",
                    "topology_metadata": {},
                    "time_delta": "",
                    "travel_window": {},
                    "source_zone_id": "",
                    "target_zone_id": event.get("zone_id", ""),
                    "zone_valid": bool(event.get("zone_id")),
                    "zone_reason": event.get("zone_reason", ""),
                    "fallback_without_zone": bool(event.get("zone_fallback_used", False)),
                    "modality_primary": "face",
                    "modality_secondary": "",
                    "face_score": round(known_match["score"], 4),
                    "body_score": "",
                    "thresholds_used": known_audit["thresholds_used"],
                    "margin_used": {
                        "actual_margin": round(known_match["margin"], 4),
                        "required_margin": float(decision_cfg["known_margin_threshold"]),
                    },
                    "decision": "known_accept",
                    "reason_code": "known_accept",
                    "gallery_id_before": "",
                    "gallery_id_after": known_match["identity_id"],
                    "candidate_evaluations": [],
                }
            )
            continue

        quality = evaluate_quality_gate(item, merged_policy["quality_gate"])
        candidate_scores = []
        for profile in profiles:
            candidate = evaluate_profile_candidate(item, profile, topology, merged_policy)
            allowed, acceptance_reason, threshold_info = _candidate_acceptance(candidate, decision_cfg)
            candidate["acceptance_pass"] = allowed
            candidate["acceptance_reason"] = acceptance_reason
            candidate["thresholds_used"] = threshold_info
            candidate["ranking_key"] = json.dumps(_ranking_key(candidate))
            candidate_scores.append(candidate)
        candidate_scores.sort(key=_ranking_key, reverse=True)

        top1 = candidate_scores[0] if candidate_scores else None
        top2 = candidate_scores[1] if len(candidate_scores) > 1 else None
        selected = None
        margin = _primary_margin(top1, top2) if top1 else 0.0
        required_margin = 0.0
        selected_thresholds = {}

        if quality["gate_pass"] and top1 and top1["acceptance_pass"]:
            required_margin = float(
                decision_cfg["margin_by_relation"].get(
                    top1["relation_type"],
                    decision_cfg["margin_by_relation"]["weak_link"],
                )
            )
            selected_thresholds = top1.get("thresholds_used", {})
            if margin >= required_margin:
                selected = top1
                selected["decision_reason"] = "unknown_reuse"
                selected["reason_code"] = "unknown_reuse"
            else:
                top1["decision_reason"] = "ambiguous_margin_reject"
                top1["reason_code"] = "ambiguous"

        if selected is not None:
            profile = next(profile for profile in profiles if profile["unknown_global_id"] == selected["candidate_unknown_global_id"])
            update_unknown_profile(profile, item, policy=gallery_cfg)
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
            decision_logs.append(
                _build_decision_log(
                    event,
                    quality,
                    candidate_scores,
                    decision="unknown_reuse",
                    reason_code="unknown_reuse",
                    selected_candidate=selected,
                    gallery_id_before=selected["candidate_unknown_global_id"],
                    gallery_id_after=profile["unknown_global_id"],
                    thresholds_used={**selected_thresholds, "required_margin": required_margin},
                    margin_used={"actual_margin": margin, "required_margin": required_margin},
                )
            )
            continue

        if not quality["gate_pass"]:
            action = decision_cfg["defer_policy"]["quality_gate_fail_action"]
            if action != "defer":
                action = "defer"
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
                        "event_zone_id": event.get("zone_id", ""),
                        "event_zone_type": event.get("zone_type", ""),
                        "candidate_unknown_global_id": "",
                        "candidate_latest_camera": "",
                        "candidate_latest_time": "",
                        "profile_camera": "",
                        "profile_time": "",
                        "source_zone_id": "",
                        "target_zone_id": event.get("zone_id", ""),
                        "relation_type": "",
                        "same_area_overlap": "",
                        "transition_rule_used": "",
                        "min_travel_time": "",
                        "avg_travel_time": "",
                        "max_travel_time": "",
                        "travel_window": {},
                        "delta_sec": "",
                        "topology_valid": "",
                        "time_valid": "",
                        "zone_valid": "",
                        "topology_allowed": "",
                        "zone_allowed": "",
                        "time_score": "",
                        "topology_score": "",
                        "zone_score": "",
                        "zone_reason": event.get("zone_reason", ""),
                        "time_reason": "",
                        "rejection_reason": "",
                        "fallback_without_zone": bool(event.get("zone_fallback_used", False)),
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
            decision_logs.append(
                _build_decision_log(
                    event,
                    quality,
                    candidate_scores,
                    decision=action,
                    reason_code=quality["reason_code"],
                    selected_candidate=None,
                    gallery_id_before="",
                    gallery_id_after="",
                    thresholds_used={"defer_policy": decision_cfg["defer_policy"]},
                    margin_used={"actual_margin": margin, "required_margin": ""},
                )
            )
            continue

        unknown_global_id = f"{unknown_prefix}_{next_unknown:04d}"
        next_unknown += 1
        had_previous_profiles = bool(profiles)
        create_reason = "no_candidate"
        if top1 is not None:
            if top1.get("reason_code") == "ambiguous":
                if quality["quality_reliability"] <= float(decision_cfg["defer_policy"]["quality_reliability_max"]):
                    low_quality_action = decision_cfg["defer_policy"]["ambiguous_low_quality_action"]
                    if low_quality_action == "defer":
                        resolved_rows.append(
                            _build_resolved_row(
                                event,
                                item,
                                decision_type="defer",
                                reason_code="ambiguous_low_quality_defer",
                                resolution_source="model_defer_ambiguous_low_quality",
                            )
                        )
                        for candidate in candidate_scores:
                            decision_reason = candidate.get("decision_reason") or "ambiguous_low_quality_defer"
                            trace_rows.append(
                                _build_trace_row(
                                    event,
                                    candidate,
                                    decision_reason=decision_reason,
                                    reused_old_id=False,
                                    created_new_id=False,
                                    selected_candidate=False,
                                )
                            )
                        decision_logs.append(
                            _build_decision_log(
                                event,
                                quality,
                                candidate_scores,
                                decision="defer",
                                reason_code="ambiguous_low_quality_defer",
                                selected_candidate=top1,
                                gallery_id_before="",
                                gallery_id_after="",
                                thresholds_used={**top1.get("thresholds_used", {}), "defer_policy": decision_cfg["defer_policy"]},
                                margin_used={
                                    "actual_margin": margin,
                                    "required_margin": float(
                                        decision_cfg["margin_by_relation"].get(
                                            top1["relation_type"],
                                            decision_cfg["margin_by_relation"]["weak_link"],
                                        )
                                    ),
                                },
                            )
                        )
                        continue
                    create_reason = "ambiguous_low_quality_create"
                else:
                    create_reason = "ambiguous_create_unknown"
            else:
                create_reason = top1.get("acceptance_reason") or top1.get("candidate_reason") or "below_threshold"
        elif not had_previous_profiles:
            create_reason = "no_previous_unknown_profiles"
        profile = create_unknown_profile(unknown_global_id, item, policy=gallery_cfg)
        profiles.append(profile)
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
                    "event_zone_id": event.get("zone_id", ""),
                    "event_zone_type": event.get("zone_type", ""),
                    "candidate_unknown_global_id": "",
                    "candidate_latest_camera": "",
                    "candidate_latest_time": "",
                    "profile_camera": "",
                    "profile_time": "",
                    "source_zone_id": "",
                    "target_zone_id": event.get("zone_id", ""),
                    "relation_type": "",
                    "same_area_overlap": "",
                    "transition_rule_used": "",
                    "min_travel_time": "",
                    "avg_travel_time": "",
                    "max_travel_time": "",
                    "travel_window": {},
                    "delta_sec": "",
                    "topology_valid": "",
                    "time_valid": "",
                    "zone_valid": "",
                    "topology_allowed": "",
                    "zone_allowed": "",
                    "time_score": "",
                    "topology_score": "",
                    "zone_score": "",
                    "zone_reason": event.get("zone_reason", ""),
                    "time_reason": "",
                    "rejection_reason": "",
                    "fallback_without_zone": bool(event.get("zone_fallback_used", False)),
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
        decision_logs.append(
            _build_decision_log(
                event,
                quality,
                candidate_scores,
                decision="create_new",
                reason_code=create_reason,
                selected_candidate=top1,
                gallery_id_before=top1["candidate_unknown_global_id"] if top1 else "",
                gallery_id_after=unknown_global_id,
                thresholds_used={
                    **(top1.get("thresholds_used", {}) if top1 else {}),
                    "create_rule": decision_cfg["create_rule"],
                },
                margin_used={
                    "actual_margin": margin,
                    "required_margin": float(
                        decision_cfg["margin_by_relation"].get(
                            top1["relation_type"],
                            decision_cfg["margin_by_relation"]["weak_link"],
                        )
                    )
                    if top1
                    else "",
                },
            )
        )
    debug_bundle = {"policy": merged_policy, "decision_logs": decision_logs}
    if return_debug_bundle:
        return resolved_rows, profiles, trace_rows, debug_bundle
    return resolved_rows, profiles, trace_rows
