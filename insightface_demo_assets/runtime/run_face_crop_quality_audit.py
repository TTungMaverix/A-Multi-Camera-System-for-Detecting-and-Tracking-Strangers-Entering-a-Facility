import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Audit face crop quality from existing evaluation artifacts.")
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile_median(values):
    if not values:
        return 0
    return statistics.median(values)


def summarize_series(values):
    if not values:
        return {"min": 0, "median": 0, "max": 0, "count": 0}
    return {
        "min": min(values),
        "median": percentile_median(values),
        "max": max(values),
        "count": len(values),
    }


def main():
    args = parse_args()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    output_json = Path(args.output_json).resolve()

    audit_csv_paths = sorted(evaluation_dir.glob("offline_runs/*/audit/audit_face_buffer.csv"))
    if not audit_csv_paths:
        payload = {
            "status": "FACE_CROP_SIZE_NOT_MEASURED",
            "evaluation_dir": str(evaluation_dir),
            "missing_files": ["offline_runs/*/audit/audit_face_buffer.csv"],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    rows = []
    for audit_csv_path in audit_csv_paths:
        pair_id = audit_csv_path.parents[1].name
        with audit_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_pair_id"] = pair_id
                rows.append(row)

    total_face_candidates = 0
    selected_best_shots = 0
    embedding_created_count = 0
    buffer_embedding_created_count = 0
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    reject_reason_counts = Counter()
    camera_breakdown = defaultdict(lambda: {"rows": 0, "face_candidates": 0, "selected_best_shots": 0, "embedding_created_count": 0})
    pair_breakdown = defaultdict(lambda: {"rows": 0, "face_candidates": 0, "selected_best_shots": 0, "embedding_created_count": 0})
    smallest_examples = []

    for row in rows:
        pair_id = row["_pair_id"]
        camera_id = row.get("camera_id", "")
        face_candidates = as_int(row.get("face_buffer_detected_count", 0))
        selected = str(row.get("face_best_shot_selected", "")).strip().lower() == "true"
        embedding_count = as_int(row.get("face_buffer_embedding_count", 0))
        bbox_width = as_int(row.get("face_bbox_width", 0))
        bbox_height = as_int(row.get("face_bbox_height", 0))
        bbox_area = as_int(row.get("face_bbox_area", 0))

        total_face_candidates += face_candidates
        selected_best_shots += 1 if selected else 0
        buffer_embedding_created_count += embedding_count
        if selected and embedding_count > 0:
            embedding_created_count += 1

        camera_breakdown[camera_id]["rows"] += 1
        camera_breakdown[camera_id]["face_candidates"] += face_candidates
        camera_breakdown[camera_id]["selected_best_shots"] += 1 if selected else 0
        camera_breakdown[camera_id]["embedding_created_count"] += 1 if (selected and embedding_count > 0) else 0
        camera_breakdown[camera_id]["buffer_embedding_created_count"] = (
            camera_breakdown[camera_id].get("buffer_embedding_created_count", 0) + embedding_count
        )

        pair_breakdown[pair_id]["rows"] += 1
        pair_breakdown[pair_id]["face_candidates"] += face_candidates
        pair_breakdown[pair_id]["selected_best_shots"] += 1 if selected else 0
        pair_breakdown[pair_id]["embedding_created_count"] += 1 if (selected and embedding_count > 0) else 0
        pair_breakdown[pair_id]["buffer_embedding_created_count"] = (
            pair_breakdown[pair_id].get("buffer_embedding_created_count", 0) + embedding_count
        )

        reject_reason_counts["size_reject"] += as_int(row.get("face_buffer_reject_size_count", 0))
        reject_reason_counts["blur_reject"] += as_int(row.get("face_buffer_reject_blur_count", 0))
        reject_reason_counts["yaw_reject"] += as_int(row.get("face_buffer_reject_yaw_count", 0))
        reject_reason_counts["pitch_reject"] += as_int(row.get("face_buffer_reject_pitch_count", 0))
        reject_reason_counts["roll_reject"] += as_int(row.get("face_buffer_reject_roll_count", 0))
        reject_reason_counts["missing_landmarks_reject"] += as_int(row.get("face_buffer_reject_missing_landmarks_count", 0))
        reject_reason_counts["camera_disabled_reject"] += as_int(row.get("face_buffer_reject_camera_disabled_count", 0))

        gate_reason = str(row.get("face_gate_reject_reason", "") or "").strip()
        if gate_reason:
            reject_reason_counts[f"gate:{gate_reason}"] += 1

        if bbox_width > 0 and bbox_height > 0 and bbox_area > 0:
            bbox_widths.append(bbox_width)
            bbox_heights.append(bbox_height)
            bbox_areas.append(bbox_area)
            smallest_examples.append(
                {
                    "pair_id": pair_id,
                    "camera_id": camera_id,
                    "event_id": row.get("event_id", ""),
                    "used_face_crop_path": row.get("used_face_crop_path", ""),
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_area": bbox_area,
                    "face_status": row.get("face_status", ""),
                    "face_gate_reject_reason": gate_reason,
                }
            )

    smallest_examples.sort(key=lambda item: (item["bbox_area"], item["bbox_width"], item["bbox_height"]))
    payload = {
        "status": "ok" if bbox_areas else "FACE_CROP_SIZE_NOT_MEASURED",
        "evaluation_dir": str(evaluation_dir),
        "audit_face_buffer_files": [str(path) for path in audit_csv_paths],
        "total_face_candidates": total_face_candidates,
        "selected_best_shots": selected_best_shots,
        "embedding_created_count": embedding_created_count,
        "buffer_embedding_created_count": buffer_embedding_created_count,
        "bbox_width_stats": summarize_series(bbox_widths),
        "bbox_height_stats": summarize_series(bbox_heights),
        "bbox_area_stats": summarize_series(bbox_areas),
        "reject_reason_counts": dict(sorted(reject_reason_counts.items())),
        "camera_id_breakdown": dict(sorted(camera_breakdown.items())),
        "pair_id_breakdown": dict(sorted(pair_breakdown.items())),
        "examples_smallest_candidates": smallest_examples[:10],
    }
    if not bbox_areas:
        payload["missing_fields"] = [
            "face_bbox_width",
            "face_bbox_height",
            "face_bbox_area",
        ]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
