import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from association_core.body_reid import load_image_unicode
from run_body_tracklet_evaluation import (
    comparison_rows_for_run,
    evaluate_run,
    merge_policy_variant,
    parse_event_buffer,
    read_csv_rows,
)
from association_core.config_loader import load_association_policy


A3_CV_VARIANTS = (
    {
        "variant_id": "baseline_current",
        "description": "Current active body preprocessing config from the association policy.",
        "body_reid_override": {},
    },
    {
        "variant_id": "no_preproc_no_shrink",
        "description": "No traditional color preprocessing and no bbox shrink.",
        "body_reid_override": {
            "preprocessing_mode": "none",
            "clahe_enabled": False,
            "gray_world_normalization": False,
            "bbox_shrink_ratio": 0.0,
        },
    },
    {
        "variant_id": "clahe_only",
        "description": "CLAHE-only preprocessing without bbox shrink.",
        "body_reid_override": {
            "preprocessing_mode": "clahe",
            "clahe_enabled": True,
            "gray_world_normalization": False,
            "bbox_shrink_ratio": 0.0,
        },
    },
    {
        "variant_id": "gray_world_only",
        "description": "Gray World color constancy without bbox shrink.",
        "body_reid_override": {
            "preprocessing_mode": "gray_world",
            "clahe_enabled": False,
            "gray_world_normalization": True,
            "bbox_shrink_ratio": 0.0,
        },
    },
    {
        "variant_id": "histogram_match_only",
        "description": "Histogram-match target crops to the source tracklet reference without bbox shrink.",
        "body_reid_override": {
            "preprocessing_mode": "histogram_match",
            "clahe_enabled": False,
            "gray_world_normalization": False,
            "bbox_shrink_ratio": 0.0,
        },
    },
    {
        "variant_id": "shrink_only",
        "description": "Remove 10% crop margins without color preprocessing.",
        "body_reid_override": {
            "preprocessing_mode": "none",
            "clahe_enabled": False,
            "gray_world_normalization": False,
            "bbox_shrink_ratio": 0.1,
        },
    },
    {
        "variant_id": "gray_world_shrink",
        "description": "Gray World plus 10% bbox shrink.",
        "body_reid_override": {
            "preprocessing_mode": "gray_world",
            "clahe_enabled": False,
            "gray_world_normalization": True,
            "bbox_shrink_ratio": 0.1,
        },
    },
    {
        "variant_id": "histogram_match_shrink",
        "description": "Histogram matching plus 10% bbox shrink.",
        "body_reid_override": {
            "preprocessing_mode": "histogram_match",
            "clahe_enabled": False,
            "gray_world_normalization": False,
            "bbox_shrink_ratio": 0.1,
        },
    },
)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_if_exists(source_path: Path, destination_path: Path):
    if source_path.exists():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _camera_bucket(camera_id):
    return "c1" if str(camera_id or "").strip() in {"C1", "C3"} else "c2"


def _stack_labeled_pair(left_path: Path, right_path: Path, label):
    left = load_image_unicode(left_path)
    right = load_image_unicode(right_path)
    if left is None or right is None:
        return None
    target_height = max(left.shape[0], right.shape[0])
    if left.shape[0] != target_height:
        left = cv2.copyMakeBorder(left, 0, target_height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    if right.shape[0] != target_height:
        right = cv2.copyMakeBorder(right, 0, target_height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    canvas = cv2.hconcat([left, right])
    canvas = cv2.copyMakeBorder(canvas, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(canvas, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    return canvas


def _write_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode {path}")
    encoded.tofile(str(path))


def _contact_sheet(images, columns=2, background=(18, 18, 18)):
    if not images:
        return None
    max_width = max(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)
    normalized = []
    for image in images:
        canvas = np.full((max_height, max_width, 3), background, dtype=np.uint8)
        offset_x = (max_width - image.shape[1]) // 2
        offset_y = (max_height - image.shape[0]) // 2
        canvas[offset_y : offset_y + image.shape[0], offset_x : offset_x + image.shape[1]] = image
        normalized.append(canvas)
    rows = []
    for index in range(0, len(normalized), columns):
        row_images = normalized[index : index + columns]
        while len(row_images) < columns:
            row_images.append(np.full((max_height, max_width, 3), background, dtype=np.uint8))
        rows.append(cv2.hconcat(row_images))
    return cv2.vconcat(rows)


def _lab_mean(image_path: Path):
    image = load_image_unicode(image_path)
    if image is None:
        return [0.0, 0.0, 0.0]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    means = lab.reshape(-1, 3).mean(axis=0)
    return [round(float(value), 3) for value in means]


def _pair_visual_metrics(source_path: Path, target_path: Path):
    source_lab = _lab_mean(source_path)
    target_lab = _lab_mean(target_path)
    delta = round(float(np.linalg.norm(np.asarray(source_lab) - np.asarray(target_lab))), 3)
    source_image = load_image_unicode(source_path)
    target_image = load_image_unicode(target_path)
    return {
        "source_lab_mean": source_lab,
        "target_lab_mean": target_lab,
        "lab_mean_distance": delta,
        "source_shape": list(source_image.shape[:2]) if source_image is not None else [],
        "target_shape": list(target_image.shape[:2]) if target_image is not None else [],
    }


def _variant_scores_from_rows(rows, variant_ids):
    summary = {}
    for variant_id in variant_ids:
        scores = [float(row.get(f"{variant_id}_body_score", 0.0) or 0.0) for row in rows]
        summary[variant_id] = {
            "average_body_score": round(float(np.mean(scores)) if scores else 0.0, 4),
            "max_body_score": round(float(np.max(scores)) if scores else 0.0, 4),
            "min_body_score": round(float(np.min(scores)) if scores else 0.0, 4),
        }
    return summary


def _write_report(output_path: Path, pair_id, comparison_rows, visual_rows, preprocessing_summary, shrink_summary):
    lines = [
        f"# {pair_id} Hard-Case Report",
        "",
        f"- comparison_count: `{len(comparison_rows)}`",
        "",
        "## Visual Notes",
        "",
    ]
    for row in visual_rows:
        lines.append(
            f"- `{row['source_camera_id']}->{row['target_camera_id']}` "
            f"lab_mean_distance=`{row['lab_mean_distance']}` "
            f"source_shape=`{row['source_shape']}` target_shape=`{row['target_shape']}`"
        )
    lines.extend(
        [
            "",
            "## Preprocessing Benchmark",
            "",
            f"- preprocessing_summary: `{preprocessing_summary}`",
            "",
            "## BBox Shrink Benchmark",
            "",
            f"- bbox_shrink_summary: `{shrink_summary}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze hard-case a3 with traditional CV preprocessing and bbox shrink.")
    parser.add_argument("--run-output-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-id", default="a3")
    parser.add_argument(
        "--association-policy-config",
        default="insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml",
    )
    args = parser.parse_args()

    run_root = Path(args.run_output_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, _runtime = load_association_policy(args.association_policy_config, base_dir=Path.cwd())
    event_rows = read_csv_rows(run_root / "runtime" / "generated_candidate_events_mode_b.csv")
    event_by_id = {row["event_id"]: row for row in event_rows}
    comparison_baseline = comparison_rows_for_run(run_root, policy)

    evaluation_dir = output_dir / "evaluation"
    summary = evaluate_run(
        run_root,
        evaluation_dir,
        args.association_policy_config,
        variants=A3_CV_VARIANTS,
    )
    comparison_rows = list(summary.get("rows", []) or [])

    crops_root = output_dir / "a3_crops"
    for row in comparison_rows:
        _copy_if_exists(Path(row["source_best_body_crop"]), crops_root / _camera_bucket(row["source_camera_id"]) / Path(row["source_best_body_crop"]).name)
        _copy_if_exists(Path(row["target_best_body_crop"]), crops_root / _camera_bucket(row["target_camera_id"]) / Path(row["target_best_body_crop"]).name)
        source_event = event_by_id.get(row["source_event_id"], {})
        target_event = event_by_id.get(row["observation_event_id"], {})
        for buffer_row in parse_event_buffer(source_event):
            _copy_if_exists(Path(buffer_row.get("body_crop_path", "")), crops_root / _camera_bucket(row["source_camera_id"]) / Path(buffer_row.get("body_crop_path", "")).name)
        for buffer_row in parse_event_buffer(target_event):
            _copy_if_exists(Path(buffer_row.get("body_crop_path", "")), crops_root / _camera_bucket(row["target_camera_id"]) / Path(buffer_row.get("body_crop_path", "")).name)

    contact_images = []
    visual_rows = []
    for row in comparison_rows:
        label = (
            f"{row['source_camera_id']}->{row['target_camera_id']} "
            f"baseline={row.get('baseline_current_body_score', 0.0)} "
            f"gray_world={row.get('gray_world_only_body_score', 0.0)} "
            f"shrink={row.get('shrink_only_body_score', 0.0)}"
        )
        stacked = _stack_labeled_pair(Path(row["source_best_body_crop"]), Path(row["target_best_body_crop"]), label)
        if stacked is not None:
            contact_images.append(stacked)
        metrics = _pair_visual_metrics(Path(row["source_best_body_crop"]), Path(row["target_best_body_crop"]))
        metrics["source_camera_id"] = row["source_camera_id"]
        metrics["target_camera_id"] = row["target_camera_id"]
        visual_rows.append(metrics)
    contact_sheet = _contact_sheet(contact_images, columns=1)
    if contact_sheet is not None:
        _write_image(output_dir / "a3_contact_sheet.png", contact_sheet)

    preprocessing_variants = [
        "baseline_current",
        "no_preproc_no_shrink",
        "clahe_only",
        "gray_world_only",
        "histogram_match_only",
    ]
    shrink_variants = [
        "no_preproc_no_shrink",
        "shrink_only",
        "gray_world_only",
        "gray_world_shrink",
        "histogram_match_only",
        "histogram_match_shrink",
    ]
    preprocessing_summary = _variant_scores_from_rows(comparison_rows, preprocessing_variants)
    shrink_summary = _variant_scores_from_rows(comparison_rows, shrink_variants)

    save_json(output_dir / "a3_preprocessing_benchmark.json", preprocessing_summary)
    save_json(output_dir / "a3_bbox_shrink_benchmark.json", shrink_summary)
    save_json(output_dir / "a3_visual_metrics.json", visual_rows)
    _write_report(
        output_dir / "a3_hard_case_report.md",
        args.pair_id,
        comparison_rows,
        visual_rows,
        preprocessing_summary,
        shrink_summary,
    )

    print(output_dir / "a3_preprocessing_benchmark.json")


if __name__ == "__main__":
    main()
