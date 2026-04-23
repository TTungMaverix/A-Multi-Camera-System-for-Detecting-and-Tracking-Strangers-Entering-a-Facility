import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from association_core.appearance_evidence import evaluate_appearance_evidence
from association_core.body_reid import build_tracklet_body_representation, get_body_reid_extractor
from association_core.config_loader import deep_merge, load_association_policy


DEFAULT_VARIANTS = (
    {
        "variant_id": "osnet_mean",
        "description": "Current OSNet extractor with mean pooling over the selected tracklet crops.",
        "body_reid_override": {
            "tracklet_pooling_mode": "mean",
        },
    },
    {
        "variant_id": "osnet_quality_aware",
        "description": "Current OSNet extractor with quality-aware weighted pooling over the selected tracklet crops.",
        "body_reid_override": {
            "tracklet_pooling_mode": "quality_aware",
        },
    },
    {
        "variant_id": "osnet_x1_0_quality_aware",
        "description": "Stronger OSNet x1.0 extractor with quality-aware weighted pooling.",
        "body_reid_override": {
            "extractor_name": "osnet_x1_0",
            "tracklet_pooling_mode": "quality_aware",
        },
    },
)


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_image_unicode(path: Path):
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    encoded.tofile(str(path))


def _stack_body_crops(source_path: Path, target_path: Path, label_text):
    left = load_image_unicode(source_path)
    right = load_image_unicode(target_path)
    if left is None or right is None:
        return None
    height = max(left.shape[0], right.shape[0])
    left = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    right = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    canvas = cv2.hconcat([left, right])
    canvas = cv2.copyMakeBorder(canvas, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(canvas, label_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
    return canvas


def merge_policy_variant(base_policy, body_reid_override):
    variant_policy = deepcopy(base_policy)
    variant_policy["body_reid"] = deep_merge(base_policy.get("body_reid", {}), body_reid_override or {})
    return variant_policy


def relation_primary_threshold(policy, relation_type):
    decision_cfg = (policy.get("decision_policy", {}) or {})
    relation_thresholds = (decision_cfg.get("relation_thresholds", {}) or {})
    relation_cfg = relation_thresholds.get(relation_type, relation_thresholds.get("weak_link", {}))
    primary_floor = float(decision_cfg.get("unknown_reuse_threshold", 0.0) or 0.0)
    return max(primary_floor, float((relation_cfg or {}).get("body_primary", primary_floor) or primary_floor))


def parse_event_buffer(event_row):
    raw_value = event_row.get("evidence_buffer_json", "[]")
    if not raw_value:
        return []
    try:
        rows = json.loads(raw_value)
    except json.JSONDecodeError:
        rows = []
    return rows or []


def synthetic_body_score(source_tracklet, target_tracklet, policy):
    if source_tracklet.get("embedding") is None or target_tracklet.get("embedding") is None:
        return 0.0
    appearance = evaluate_appearance_evidence(
        {
            "face_embedding": None,
            "body_embedding": target_tracklet.get("embedding"),
            "body_tracklet_embeddings": target_tracklet.get("tracklet_embeddings", []),
        },
        {
            "body_refs": [
                {
                    "embedding": source_tracklet.get("embedding"),
                    "tracklet_embeddings": source_tracklet.get("tracklet_embeddings", []),
                }
            ],
            "representative_body_embedding": source_tracklet.get("embedding"),
            "face_refs": [],
            "representative_face_embedding": None,
        },
        {
            "appearance_evidence_policy": policy.get("appearance_evidence", {}),
            "face_reliable": False,
            "body_reliable": True,
            "primary_modality": "body",
        },
    )
    return round(float(appearance.get("body_score", 0.0) or 0.0), 4)


def _extract_single_frame_embedding(extractor, image_path: Path):
    result = extractor.extract(image_path)
    return result.get("embedding"), result.get("status", "missing")


def _single_frame_body_score(source_path: Path, target_path: Path, policy):
    extractor = get_body_reid_extractor(policy=policy.get("body_reid"))
    source_embedding, source_status = _extract_single_frame_embedding(extractor, source_path)
    target_embedding, target_status = _extract_single_frame_embedding(extractor, target_path)
    if source_embedding is None or target_embedding is None:
        return 0.0, {
            "source_status": source_status,
            "target_status": target_status,
        }
    appearance = evaluate_appearance_evidence(
        {
            "face_embedding": None,
            "body_embedding": target_embedding,
            "body_tracklet_embeddings": [],
        },
        {
            "body_refs": [{"embedding": source_embedding, "tracklet_embeddings": []}],
            "representative_body_embedding": source_embedding,
            "face_refs": [],
            "representative_face_embedding": None,
        },
        {
            "appearance_evidence_policy": policy.get("appearance_evidence", {}),
            "face_reliable": False,
            "body_reliable": True,
            "primary_modality": "body",
        },
    )
    return round(float(appearance.get("body_score", 0.0) or 0.0), 4), {
        "source_status": source_status,
        "target_status": target_status,
    }


def _top_candidate_from_log(log_row):
    candidates = list(log_row.get("candidate_evaluations", []) or [])
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            1 if row.get("topology_allowed") else 0,
            float(row.get("appearance_primary", 0.0) or 0.0),
            float(row.get("appearance_secondary", 0.0) or 0.0),
            float(row.get("time_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return candidates[0]


def _source_event_for_candidate(candidate_unknown_id, source_camera_id, target_relative_sec, resolved_rows, event_by_id):
    if not candidate_unknown_id or not source_camera_id:
        return None
    candidates = [
        row
        for row in resolved_rows
        if row.get("resolved_global_id") == candidate_unknown_id
        and row.get("camera_id") == source_camera_id
        and float(row.get("relative_sec") or 0.0) <= float(target_relative_sec or 0.0)
    ]
    candidates.sort(key=lambda row: float(row.get("relative_sec") or 0.0), reverse=True)
    if not candidates:
        return None
    return event_by_id.get(candidates[0].get("event_id", ""))


def build_variant_tracklet(source_event, target_event, variant_policy):
    extractor = get_body_reid_extractor(policy=variant_policy.get("body_reid"))
    source_tracklet = build_tracklet_body_representation(
        parse_event_buffer(source_event),
        body_reid_runtime=extractor,
        body_reid_policy=variant_policy.get("body_reid"),
    )
    target_tracklet = build_tracklet_body_representation(
        parse_event_buffer(target_event),
        body_reid_runtime=extractor,
        body_reid_policy=variant_policy.get("body_reid"),
    )
    score = synthetic_body_score(source_tracklet, target_tracklet, variant_policy)
    return {
        "body_score": score,
        "source_tracklet": source_tracklet,
        "target_tracklet": target_tracklet,
    }


def comparison_rows_for_run(run_root: Path, policy):
    decision_logs = read_jsonl(run_root / "association_logs" / "association_decisions.jsonl")
    if not decision_logs:
        decision_logs = read_jsonl(run_root / "runtime" / "association_logs" / "association_decisions.jsonl")
    resolved_rows = read_csv_rows(run_root / "events" / "resolved_events.csv")
    event_rows = read_csv_rows(run_root / "runtime" / "generated_candidate_events_mode_b.csv")
    event_by_id = {row["event_id"]: row for row in event_rows}

    comparison_rows = []
    for log in decision_logs:
        if log.get("camera_id") == log.get("source_camera_id"):
            continue
        target_event = event_by_id.get(log.get("observation_id", ""))
        if not target_event:
            continue
        top_candidate = _top_candidate_from_log(log)
        source_camera_id = str(log.get("source_camera_id") or (top_candidate or {}).get("source_camera_id") or "").strip()
        candidate_unknown_id = str(log.get("selected_candidate_id") or (top_candidate or {}).get("candidate_unknown_global_id") or "").strip()
        source_event = _source_event_for_candidate(
            candidate_unknown_id,
            source_camera_id,
            float(log.get("relative_time") or 0.0),
            resolved_rows,
            event_by_id,
        )
        if source_event is None:
            continue

        relation_type = str(log.get("relation_type") or (top_candidate or {}).get("relation_type") or "weak_link")
        primary_threshold = relation_primary_threshold(policy, relation_type)
        source_best = Path(source_event.get("best_body_crop", ""))
        target_best = Path(target_event.get("best_body_crop", ""))
        old_score, single_frame_status = _single_frame_body_score(source_best, target_best, policy)
        comparison_rows.append(
            {
                "observation_event_id": target_event["event_id"],
                "source_event_id": source_event.get("event_id", ""),
                "source_camera_id": source_event.get("camera_id", ""),
                "target_camera_id": target_event.get("camera_id", ""),
                "decision": log.get("decision", ""),
                "final_reason_code": log.get("reason_code", ""),
                "selected_candidate_id": candidate_unknown_id,
                "relation_type": relation_type,
                "primary_threshold": round(primary_threshold, 4),
                "topology_support_level": (top_candidate or {}).get("topology_support_level", ""),
                "acceptance_reason": (top_candidate or {}).get("acceptance_reason", ""),
                "appearance_primary_logged": round(float((top_candidate or {}).get("appearance_primary", 0.0) or 0.0), 4),
                "appearance_secondary_logged": round(float((top_candidate or {}).get("appearance_secondary", 0.0) or 0.0), 4),
                "old_single_frame_body_score": old_score,
                "single_frame_source_status": single_frame_status["source_status"],
                "single_frame_target_status": single_frame_status["target_status"],
                "source_best_body_crop": str(source_best),
                "target_best_body_crop": str(target_best),
                "source_buffer_count": len(parse_event_buffer(source_event)),
                "target_buffer_count": len(parse_event_buffer(target_event)),
            }
        )
    return comparison_rows


def summarize_variant_scores(comparison_rows, variants):
    variant_metrics = {}
    for variant in variants:
        scores = [float(row.get(f"{variant['variant_id']}_body_score", 0.0) or 0.0) for row in comparison_rows]
        passes = [row for row in comparison_rows if row.get(f"{variant['variant_id']}_appearance_only_pass")]
        variant_metrics[variant["variant_id"]] = {
            "description": variant["description"],
            "average_body_score": round(float(np.mean(scores)) if scores else 0.0, 4),
            "max_body_score": round(float(np.max(scores)) if scores else 0.0, 4),
            "min_body_score": round(float(np.min(scores)) if scores else 0.0, 4),
            "appearance_only_pass_count": len(passes),
        }
    return variant_metrics


def evaluate_run(run_root: Path, output_dir: Path, association_policy_config: str, variants=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    policy, runtime = load_association_policy(association_policy_config, base_dir=Path.cwd())
    variants = list(variants or DEFAULT_VARIANTS)
    comparison_rows = comparison_rows_for_run(run_root, policy)

    for row in comparison_rows:
        source_event = {
            "event_id": row["source_event_id"],
            "camera_id": row["source_camera_id"],
            "best_body_crop": row["source_best_body_crop"],
        }
        target_event = {
            "event_id": row["observation_event_id"],
            "camera_id": row["target_camera_id"],
            "best_body_crop": row["target_best_body_crop"],
        }
        # Reload the original event rows to preserve evidence buffers.
        event_rows = read_csv_rows(run_root / "runtime" / "generated_candidate_events_mode_b.csv")
        event_by_id = {item["event_id"]: item for item in event_rows}
        source_event = event_by_id.get(row["source_event_id"], source_event)
        target_event = event_by_id.get(row["observation_event_id"], target_event)

        for variant in variants:
            variant_policy = merge_policy_variant(policy, variant.get("body_reid_override"))
            variant_eval = build_variant_tracklet(source_event, target_event, variant_policy)
            score = round(float(variant_eval["body_score"]), 4)
            row[f"{variant['variant_id']}_body_score"] = score
            row[f"{variant['variant_id']}_score_delta_vs_single"] = round(score - float(row["old_single_frame_body_score"]), 4)
            row[f"{variant['variant_id']}_appearance_only_pass"] = score >= float(row["primary_threshold"])
            row[f"{variant['variant_id']}_source_selected_crop_count"] = variant_eval["source_tracklet"].get("tracklet_selected_crop_count", 0)
            row[f"{variant['variant_id']}_target_selected_crop_count"] = variant_eval["target_tracklet"].get("tracklet_selected_crop_count", 0)
            row[f"{variant['variant_id']}_source_quality_score"] = variant_eval["source_tracklet"].get("tracklet_quality_score", 0.0)
            row[f"{variant['variant_id']}_target_quality_score"] = variant_eval["target_tracklet"].get("tracklet_quality_score", 0.0)
            row[f"{variant['variant_id']}_source_weights_json"] = json.dumps(
                variant_eval["source_tracklet"].get("tracklet_selected_crop_weights", []),
                ensure_ascii=False,
            )
            row[f"{variant['variant_id']}_target_weights_json"] = json.dumps(
                variant_eval["target_tracklet"].get("tracklet_selected_crop_weights", []),
                ensure_ascii=False,
            )
            row[f"{variant['variant_id']}_source_strategy"] = variant_eval["source_tracklet"].get("tracklet_pooling_strategy", "")
            row[f"{variant['variant_id']}_target_strategy"] = variant_eval["target_tracklet"].get("tracklet_pooling_strategy", "")

        compare_image = _stack_body_crops(
            Path(row["source_best_body_crop"]),
            Path(row["target_best_body_crop"]),
            "{}->{} single={} {}".format(
                row["source_camera_id"],
                row["target_camera_id"],
                row["old_single_frame_body_score"],
                " ".join(
                    "{}={}".format(
                        variant["variant_id"],
                        row.get(f"{variant['variant_id']}_body_score", 0.0),
                    )
                    for variant in variants
                ),
            ),
        )
        if compare_image is not None:
            write_image_unicode(output_dir / f"{row['observation_event_id']}_body_compare.png", compare_image)

    summary = {
        "run_output_root": str(run_root.resolve()),
        "association_policy_runtime": runtime,
        "comparison_count": len(comparison_rows),
        "variants": summarize_variant_scores(comparison_rows, variants),
        "average_old_single_frame_body_score": round(
            float(np.mean([row["old_single_frame_body_score"] for row in comparison_rows])) if comparison_rows else 0.0,
            4,
        ),
        "average_best_variant_body_score": round(
            max(
                (
                    metrics["average_body_score"]
                    for metrics in summarize_variant_scores(comparison_rows, variants).values()
                ),
                default=0.0,
            ),
            4,
        ),
        "rows": comparison_rows,
    }
    (output_dir / "body_tracklet_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if comparison_rows:
        with (output_dir / "body_tracklet_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare single-frame vs tracklet body evidence variants on an offline run.")
    parser.add_argument("--run-output-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--association-policy-config",
        default="insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml",
    )
    args = parser.parse_args()

    run_root = Path(args.run_output_root)
    output_dir = Path(args.output_dir) if args.output_dir else (run_root / "evaluation" / "body_tracklet")
    summary = evaluate_run(run_root, output_dir, args.association_policy_config)
    print(f"BODY_TRACKLET_COMPARISON_COUNT={summary['comparison_count']}")
    print(output_dir / "body_tracklet_comparison_summary.json")


if __name__ == "__main__":
    main()
