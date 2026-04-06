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
        if row.get("candidate_set_before_filter") and not row.get("candidate_set_after_filter"):
            summary["topology_reject_count"] += 1
        candidate_evaluations = row.get("candidate_evaluations", [])
        if any(not candidate.get("zone_valid", True) for candidate in candidate_evaluations):
            summary["zone_reject_count"] += 1
        if row.get("fallback_without_zone") or any(candidate.get("fallback_without_zone", False) for candidate in candidate_evaluations):
            summary["fallback_without_zone_count"] += 1
    return summary
