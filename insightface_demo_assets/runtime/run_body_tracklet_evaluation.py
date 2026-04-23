import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from association_core.appearance_evidence import cosine_similarity
from association_core.body_reid import build_tracklet_body_representation, get_body_reid_extractor
from association_core.config_loader import load_association_policy


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


def main():
    parser = argparse.ArgumentParser(description="Compare single-frame vs tracklet-based body evidence on an offline run.")
    parser.add_argument("--run-output-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--association-policy-config",
        default="insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml",
    )
    args = parser.parse_args()

    run_root = Path(args.run_output_root)
    output_dir = Path(args.output_dir) if args.output_dir else (run_root / "evaluation" / "body_tracklet")
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, _runtime = load_association_policy(args.association_policy_config, base_dir=Path.cwd())
    extractor = get_body_reid_extractor(policy=policy.get("body_reid"))

    decision_logs = read_jsonl(run_root / "association_logs" / "association_decisions.jsonl")
    resolved_rows = read_csv_rows(run_root / "events" / "resolved_events.csv")
    event_rows = read_csv_rows(run_root / "runtime" / "generated_candidate_events_mode_b.csv")
    event_by_id = {row["event_id"]: row for row in event_rows}

    comparison_rows = []
    for log in decision_logs:
        if log.get("decision") not in {"unknown_reuse", "create_new"}:
            continue
        if not log.get("source_camera_id"):
            continue
        target_event = event_by_id.get(log["observation_id"])
        if not target_event:
            continue
        candidate_unknown_id = log.get("selected_candidate_id") or ""
        source_event = None
        if candidate_unknown_id:
            candidates = [
                row
                for row in resolved_rows
                if row.get("resolved_global_id") == candidate_unknown_id
                and row.get("camera_id") == log.get("source_camera_id")
                and float(row.get("relative_sec") or 0.0) <= float(log.get("relative_time") or 0.0)
            ]
            candidates.sort(key=lambda row: float(row.get("relative_sec") or 0.0), reverse=True)
            if candidates:
                source_event = event_by_id.get(candidates[0]["event_id"])
        if source_event is None:
            continue

        source_best = Path(source_event.get("best_body_crop", ""))
        target_best = Path(target_event.get("best_body_crop", ""))
        single_source = extractor.extract(source_best)
        single_target = extractor.extract(target_best)
        old_score = cosine_similarity(single_source.get("embedding"), single_target.get("embedding"))

        source_tracklet = build_tracklet_body_representation(
            json.loads(source_event.get("evidence_buffer_json", "[]") or "[]"),
            body_reid_runtime=extractor,
            body_reid_policy=policy.get("body_reid"),
        )
        target_tracklet = build_tracklet_body_representation(
            json.loads(target_event.get("evidence_buffer_json", "[]") or "[]"),
            body_reid_runtime=extractor,
            body_reid_policy=policy.get("body_reid"),
        )
        new_score = cosine_similarity(source_tracklet.get("embedding"), target_tracklet.get("embedding"))
        delta = round(float(new_score) - float(old_score), 4)
        comparison_rows.append(
            {
                "target_event_id": target_event["event_id"],
                "source_camera_id": source_event.get("camera_id", ""),
                "target_camera_id": target_event.get("camera_id", ""),
                "decision": log.get("decision", ""),
                "selected_candidate_id": candidate_unknown_id,
                "old_single_frame_body_score": round(float(old_score), 4),
                "new_tracklet_body_score": round(float(new_score), 4),
                "score_delta": delta,
                "source_best_body_crop": str(source_best),
                "target_best_body_crop": str(target_best),
                "source_tracklet_selected_crop_count": source_tracklet.get("tracklet_selected_crop_count", 0),
                "target_tracklet_selected_crop_count": target_tracklet.get("tracklet_selected_crop_count", 0),
                "topology_support_level": log.get("topology_metadata", {}).get("topology_support_level", ""),
                "reason_code": log.get("reason_code", ""),
            }
        )
        compare_image = _stack_body_crops(
            source_best,
            target_best,
            f"{source_event.get('camera_id', '')}->{target_event.get('camera_id', '')} old={round(float(old_score), 4)} new={round(float(new_score), 4)}",
        )
        if compare_image is not None:
            write_image_unicode(output_dir / f"{target_event['event_id']}_body_compare.png", compare_image)

    summary = {
        "run_output_root": str(run_root.resolve()),
        "comparison_count": len(comparison_rows),
        "average_old_single_frame_body_score": round(
            float(np.mean([row["old_single_frame_body_score"] for row in comparison_rows])) if comparison_rows else 0.0,
            4,
        ),
        "average_new_tracklet_body_score": round(
            float(np.mean([row["new_tracklet_body_score"] for row in comparison_rows])) if comparison_rows else 0.0,
            4,
        ),
        "average_score_delta": round(
            float(np.mean([row["score_delta"] for row in comparison_rows])) if comparison_rows else 0.0,
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
    print(f"BODY_TRACKLET_COMPARISON_COUNT={summary['comparison_count']}")
    print(output_dir / "body_tracklet_comparison_summary.json")


if __name__ == "__main__":
    main()
