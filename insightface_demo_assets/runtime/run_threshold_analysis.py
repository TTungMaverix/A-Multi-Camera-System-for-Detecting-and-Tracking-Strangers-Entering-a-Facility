import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from evaluation_utils import read_csv_rows, save_json, write_csv_rows
from offline_pipeline.event_builder import load_json

matplotlib.use("Agg")


def resolve_path(base_dir: Path, value):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_threshold_config(config_path: Path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return payload.get("threshold_analysis", payload)


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_pair_rows(decision_logs, resolved_rows, event_gt_matches, policy):
    gt_by_event = {row["pred_event_id"]: str(row["gt_global_id"]) for row in event_gt_matches if row.get("pred_event_id")}
    unknown_gt_by_id = {}
    for row in resolved_rows:
        if row.get("identity_status") != "unknown" or not row.get("unknown_global_id"):
            continue
        actual_gt = gt_by_event.get(row.get("event_id", ""))
        if not actual_gt:
            continue
        unknown_gt_by_id.setdefault(row["unknown_global_id"], set()).add(actual_gt)

    pair_rows = []
    fusion_face_weight = float(policy["appearance_evidence"].get("fusion_face_weight", 0.7))
    fusion_body_weight = float(policy["appearance_evidence"].get("fusion_body_weight", 0.3))
    for row in decision_logs:
        event_id = row.get("observation_id", "")
        event_gt_id = gt_by_event.get(event_id)
        if not event_gt_id:
            continue
        for candidate in row.get("candidate_evaluations", []):
            if not candidate.get("hard_filter_pass", False):
                continue
            candidate_id = candidate.get("candidate_unknown_global_id", "")
            if not candidate_id:
                continue
            candidate_gt_ids = unknown_gt_by_id.get(candidate_id, set())
            if len(candidate_gt_ids) != 1:
                continue
            candidate_gt_id = next(iter(candidate_gt_ids))
            label = 1 if candidate_gt_id == event_gt_id else 0
            face_score = float(candidate.get("face_score", 0.0) or 0.0)
            body_score = float(candidate.get("body_score", 0.0) or 0.0)
            fusion_score = None
            if candidate.get("face_available") and candidate.get("body_available"):
                fusion_score = (face_score * fusion_face_weight) + (body_score * fusion_body_weight)
            pair_rows.append(
                {
                    "event_id": event_id,
                    "event_gt_id": event_gt_id,
                    "candidate_unknown_global_id": candidate_id,
                    "candidate_gt_id": candidate_gt_id,
                    "label_positive": label,
                    "relation_type": candidate.get("relation_type", ""),
                    "selected_candidate": bool(candidate.get("selected_candidate", False)),
                    "face_available": bool(candidate.get("face_available", False)),
                    "body_available": bool(candidate.get("body_available", False)),
                    "fusion_used": bool(candidate.get("fusion_used", False)),
                    "face_score": round(face_score, 6),
                    "body_score": round(body_score, 6),
                    "fusion_score": round(fusion_score, 6) if fusion_score is not None else "",
                    "appearance_primary": round(float(candidate.get("appearance_primary", 0.0) or 0.0), 6),
                    "appearance_mode": candidate.get("appearance_mode", ""),
                    "reason_code": candidate.get("reason_code", ""),
                }
            )
    return pair_rows


def _threshold_metrics(labels, scores):
    if not labels or len(set(labels)) < 2:
        return None
    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    fpr, tpr, roc_thresholds = roc_curve(labels_np, scores_np)
    roc_auc = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(labels_np, scores_np)
    pr_auc = auc(recall, precision)

    best = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    if len(pr_thresholds):
        for idx, threshold in enumerate(pr_thresholds):
            p = precision[idx + 1]
            r = recall[idx + 1]
            f1 = (2 * p * r / max(p + r, 1e-9)) if (p + r) else 0.0
            if (f1 > best["f1"]) or (math.isclose(f1, best["f1"]) and float(threshold) > best["threshold"]):
                best = {"threshold": float(threshold), "f1": float(f1), "precision": float(p), "recall": float(r)}
    positives = scores_np[labels_np == 1]
    negatives = scores_np[labels_np == 0]
    return {
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist(), "auc": round(float(roc_auc), 4)},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist(), "auc": round(float(pr_auc), 4)},
        "best_f1_threshold": {
            "threshold": round(best["threshold"], 4),
            "f1": round(best["f1"], 4),
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
        },
        "positive_count": int(len(positives)),
        "negative_count": int(len(negatives)),
        "positive_min": round(float(positives.min()), 4) if len(positives) else 0.0,
        "positive_max": round(float(positives.max()), 4) if len(positives) else 0.0,
        "positive_mean": round(float(positives.mean()), 4) if len(positives) else 0.0,
        "negative_mean": round(float(negatives.mean()), 4) if len(negatives) else 0.0,
        "negative_min": round(float(negatives.min()), 4) if len(negatives) else 0.0,
        "negative_max": round(float(negatives.max()), 4) if len(negatives) else 0.0,
        "positive_std": round(float(positives.std()), 4) if len(positives) else 0.0,
        "negative_std": round(float(negatives.std()), 4) if len(negatives) else 0.0,
    }


def analyze_modalities(pair_rows):
    modality_rows = {
        "face": [row for row in pair_rows if row["face_available"]],
        "body": [row for row in pair_rows if row["body_available"]],
        "fusion": [row for row in pair_rows if row["fusion_score"] not in ("", None)],
    }
    results = {}
    for modality, rows in modality_rows.items():
        score_key = f"{modality}_score"
        labels = [int(row["label_positive"]) for row in rows]
        scores = [float(row[score_key]) for row in rows]
        results[modality] = _threshold_metrics(labels, scores)
    return results


def plot_curves(results, pair_rows, output_root: Path):
    roc_path = output_root / "roc_curves.png"
    pr_path = output_root / "pr_curves.png"
    dist_path = output_root / "score_distributions.png"

    plt.figure(figsize=(8, 6))
    for modality, metrics in results.items():
        if not metrics:
            continue
        plt.plot(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], label=f"{modality} (AUC={metrics['roc_curve']['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Association Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    for modality, metrics in results.items():
        if not metrics:
            continue
        plt.plot(metrics["pr_curve"]["recall"], metrics["pr_curve"]["precision"], label=f"{modality} (AUC={metrics['pr_curve']['auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for Association Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, modality in zip(axes, ("face", "body", "fusion")):
        metrics = results.get(modality)
        axis.set_title(f"{modality.capitalize()} score")
        if not metrics:
            axis.text(0.5, 0.5, "insufficient data", ha="center", va="center")
            continue
        score_key = f"{modality}_score"
        positives = [float(row[score_key]) for row in pair_rows if row.get(score_key) not in ("", None) and int(row["label_positive"]) == 1]
        negatives = [float(row[score_key]) for row in pair_rows if row.get(score_key) not in ("", None) and int(row["label_positive"]) == 0]
        axis.hist(positives, bins=8, alpha=0.65, label="positive", color="#0d6b4c")
        axis.hist(negatives, bins=8, alpha=0.55, label="negative", color="#c65c2d")
        axis.set_xlim(0.0, 1.0)
        axis.legend(fontsize=8)
    fig.suptitle("Positive vs negative score distributions")
    fig.tight_layout()
    fig.savefig(dist_path, dpi=160)
    plt.close(fig)
    return {"roc_curve_png": str(roc_path), "pr_curve_png": str(pr_path), "distribution_png": str(dist_path)}


def current_thresholds_from_policy(policy):
    sequential = policy["decision_policy"]["relation_thresholds"]["sequential"]
    return {
        "face": float(sequential.get("face_primary", 0.0)),
        "body": float(sequential.get("body_primary", 0.0)),
        "fusion": float(sequential.get("fusion_primary", 0.0)),
    }


def choose_recommended_threshold(metrics, current_threshold):
    if not metrics:
        return None
    positive_min = float(metrics.get("positive_min", 0.0))
    negative_max = float(metrics.get("negative_max", 0.0))
    if metrics.get("positive_count", 0) and metrics.get("negative_count", 0) and positive_min > negative_max:
        # When positives and negatives are cleanly separated, keep the current threshold if
        # it already sits inside the safe gap. Otherwise pick the midpoint of the gap instead
        # of the smallest PR threshold, which can be overly lenient on tiny datasets.
        if negative_max < float(current_threshold) < positive_min:
            return round(float(current_threshold), 4)
        return round((positive_min + negative_max) / 2.0, 4)
    return round(float(metrics["best_f1_threshold"]["threshold"]), 4)


def main(config_path: Path):
    threshold_config = load_threshold_config(config_path)
    project_root = resolve_path(config_path.parent, threshold_config.get("project_root", "../../.."))
    benchmark_output_root = resolve_path(project_root, threshold_config["benchmark_output_root"])
    quantitative_eval_root = resolve_path(project_root, threshold_config["quantitative_eval_root"])
    output_root = resolve_path(project_root, threshold_config.get("output_root", str(benchmark_output_root / "evaluation" / "threshold_analysis")))
    output_root.mkdir(parents=True, exist_ok=True)

    decision_logs = load_jsonl(benchmark_output_root / "association_logs" / "association_decisions.jsonl")
    resolved_rows = read_csv_rows(benchmark_output_root / "events" / "resolved_events.csv")
    event_gt_matches = read_csv_rows(quantitative_eval_root / "pred_event_gt_matches.csv")
    policy_runtime = load_json(benchmark_output_root / "association_logs" / "association_policy_runtime.json")
    policy = policy_runtime["policy"]

    pair_rows = build_pair_rows(decision_logs, resolved_rows, event_gt_matches, policy)
    write_csv_rows(output_root / "candidate_pair_dataset.csv", pair_rows)
    modality_results = analyze_modalities(pair_rows)
    plot_paths = plot_curves(modality_results, pair_rows, output_root)
    current_thresholds = current_thresholds_from_policy(policy)
    recommended_thresholds = {
        modality: choose_recommended_threshold(metrics, current_thresholds.get(modality, 0.0))
        for modality, metrics in modality_results.items()
    }
    threshold_summary = {
        "benchmark_output_root": str(benchmark_output_root),
        "quantitative_eval_root": str(quantitative_eval_root),
        "pair_row_count": len(pair_rows),
        "current_sequential_thresholds": current_thresholds,
        "modality_analysis": modality_results,
        "recommended_thresholds": recommended_thresholds,
        "artifacts": plot_paths,
        "recommendation_notes": [
            "Thresholds are fitted only on hard-filter-surviving candidates, because topology/travel-time already act as a pre-similarity gate.",
            "The current replay benchmark is sequential-only, so the recommendation is most defensible for relation_thresholds.sequential.*",
            "When positives and negatives are cleanly separated on a tiny pair set, the recommendation keeps a threshold inside the safe gap instead of blindly adopting the smallest PR threshold.",
        ],
    }
    save_json(output_root / "threshold_recommendation_summary.json", threshold_summary)
    print(f"THRESHOLD_OUTPUT_ROOT={output_root}")
    for modality, threshold in recommended_thresholds.items():
        if threshold is None:
            print(f"{modality.upper()}_THRESHOLD=insufficient_data")
            continue
        print(f"{modality.upper()}_THRESHOLD={threshold}")
    return threshold_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze face/body/fusion thresholds from association candidate logs.")
    parser.add_argument("--config", required=True, help="Path to threshold analysis YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main(Path(args.config).resolve())
