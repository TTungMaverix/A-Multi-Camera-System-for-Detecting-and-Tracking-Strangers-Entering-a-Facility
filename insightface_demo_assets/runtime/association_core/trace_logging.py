import json
from pathlib import Path


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_decision_logs(decision_logs):
    summary = {
        "decision_count": len(decision_logs),
        "known_accept_count": 0,
        "unknown_reuse_count": 0,
        "new_unknown_count": 0,
        "defer_count": 0,
        "quality_gate_reject_count": 0,
        "topology_reject_count": 0,
        "zone_reject_count": 0,
        "fallback_without_zone_count": 0,
        "subzone_reject_count": 0,
        "fallback_without_subzone_count": 0,
        "body_fallback_used_count": 0,
        "face_unusable_event_count": 0,
        "pending_count": 0,
        "pending_to_reuse_count": 0,
        "pending_to_create_count": 0,
    }
    for row in decision_logs:
        decision = row.get("decision", "")
        reason_code = row.get("reason_code", "")
        if decision == "known_accept":
            summary["known_accept_count"] += 1
        elif decision == "unknown_reuse":
            summary["unknown_reuse_count"] += 1
        elif decision == "create_new":
            summary["new_unknown_count"] += 1
        elif decision == "defer":
            summary["defer_count"] += 1
        if reason_code.startswith("poor_quality"):
            summary["quality_gate_reject_count"] += 1
        candidate_evaluations = row.get("candidate_evaluations", [])
        if any((not candidate.get("topology_valid", True)) or (not candidate.get("time_valid", True)) for candidate in candidate_evaluations):
            summary["topology_reject_count"] += 1
        if any(not candidate.get("zone_valid", True) for candidate in candidate_evaluations):
            summary["zone_reject_count"] += 1
        if row.get("fallback_without_zone") or any(candidate.get("fallback_without_zone", False) for candidate in candidate_evaluations):
            summary["fallback_without_zone_count"] += 1
        if any(not candidate.get("subzone_valid", True) for candidate in candidate_evaluations):
            summary["subzone_reject_count"] += 1
        if row.get("fallback_without_subzone") or any(candidate.get("fallback_without_subzone", False) for candidate in candidate_evaluations):
            summary["fallback_without_subzone_count"] += 1
        if row.get("body_fallback_used"):
            summary["body_fallback_used_count"] += 1
        if row.get("face_unusable_reason"):
            summary["face_unusable_event_count"] += 1
        if row.get("pending_used"):
            summary["pending_count"] += 1
            if row.get("pending_resolution") == "reuse_existing_unknown":
                summary["pending_to_reuse_count"] += 1
            elif row.get("pending_resolution") == "create_new_unknown":
                summary["pending_to_create_count"] += 1
    return summary
