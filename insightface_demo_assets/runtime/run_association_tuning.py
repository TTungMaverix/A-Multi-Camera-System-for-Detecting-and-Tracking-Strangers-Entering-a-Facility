import csv
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from association_core import build_topology_index, summarize_decision_logs  # noqa: E402
from association_core.config_loader import deep_merge  # noqa: E402
from run_face_resolution_demo import (  # noqa: E402
    CONFIG_DEFAULT,
    analyze_event_crops,
    as_float,
    as_int,
    build_gallery_embeddings,
    fieldnames_for_rows,
    load_json,
    read_csv,
    save_json,
    write_csv,
)
from association_core import assign_model_identities, load_association_policy, load_camera_transition_map  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402


def resolve_path(project_root: Path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_tuning_config(config_path: Path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return payload.get("association_tuning", payload)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1", 1):
        return True
    if value in ("False", "false", "0", 0, "", None):
        return False
    return bool(value)


def parse_candidate_events(rows):
    parsed = []
    for row in rows:
        item = dict(row)
        for key in ("frame_id", "best_shot_frame", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "bbox_width", "bbox_height", "bbox_area", "anchor_frame_id"):
            if key in item:
                item[key] = as_int(item.get(key))
        for key in ("relative_sec", "best_shot_sec", "foot_x", "foot_y", "assignment_point_x", "assignment_point_y", "anchor_relative_sec", "delta_from_anchor_sec", "min_travel_time", "avg_travel_time", "max_travel_time"):
            if key in item:
                item[key] = as_float(item.get(key), 0.0 if item.get(key) not in ("", None) else "")
        for key in ("zone_fallback_used", "subzone_fallback_used", "same_area_overlap"):
            if key in item:
                item[key] = as_bool(item.get(key))
        parsed.append(item)
    return parsed


def save_feature_cache(path: Path, analyzed_events):
    payload = []
    for item in analyzed_events:
        payload.append(
            {
                "event": item["event"],
                "face_embedding": item["face_embedding"].tolist() if item["face_embedding"] is not None else None,
                "face_status": item["face_status"],
                "face_count": item["face_count"],
                "face_det_score": item["face_det_score"],
                "face_bbox": item["face_bbox"],
                "face_message": item["face_message"],
                "used_face_crop": item["used_face_crop"],
                "used_face_crop_path": item["used_face_crop_path"],
                "body_embedding": item["body_embedding"].tolist() if item["body_embedding"] is not None else None,
                "body_status": item["body_status"],
                "body_message": item["body_message"],
                "body_shape": item["body_shape"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_feature_cache(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    analyzed_events = []
    for item in payload:
        analyzed_events.append(
            {
                "event": item["event"],
                "face_embedding": np.asarray(item["face_embedding"], dtype=np.float32) if item["face_embedding"] is not None else None,
                "face_status": item["face_status"],
                "face_count": item["face_count"],
                "face_det_score": item["face_det_score"],
                "face_bbox": item["face_bbox"],
                "face_message": item["face_message"],
                "used_face_crop": item["used_face_crop"],
                "used_face_crop_path": item["used_face_crop_path"],
                "body_embedding": np.asarray(item["body_embedding"], dtype=np.float32) if item["body_embedding"] is not None else None,
                "body_status": item["body_status"],
                "body_message": item["body_message"],
                "body_shape": item["body_shape"],
            }
        )
    return analyzed_events


def load_manifest_rows(config, face_demo_config, project_root: Path):
    manifest_candidates = []
    explicit_manifest = config.get("input", {}).get("known_face_manifest_csv", "")
    if explicit_manifest:
        manifest_candidates.append(resolve_path(project_root, explicit_manifest))
    runtime_manifest = project_root / "insightface_demo_assets" / "known_face_manifest_runtime.csv"
    if runtime_manifest.exists():
        manifest_candidates.append(runtime_manifest)
    manifest_candidates.append(resolve_path(project_root, face_demo_config["known_face_manifest_csv"]))
    for candidate in manifest_candidates:
        if candidate.exists():
            return read_csv(candidate), candidate
    raise RuntimeError("No known face manifest available for tuning.")


def build_topology_for_events(wildtrack_config, transition_map):
    topology = build_topology_index(transition_map)
    fps = float(wildtrack_config["assumed_video_fps"])
    for src_camera, targets in topology.items():
        for dst_camera, info in targets.items():
            info["min_travel_time_frames"] = int(round(info["min_travel_time"] * fps))
            info["avg_travel_time_frames"] = int(round(info["avg_travel_time"] * fps))
            info["max_travel_time_frames"] = int(round(info["max_travel_time"] * fps))
    return topology


def count_true_model_based_reuse(resolved_rows):
    grouped = defaultdict(list)
    for row in resolved_rows:
        if row["identity_status"] != "unknown" or not row.get("unknown_global_id"):
            continue
        grouped[row["unknown_global_id"]].append(row)
    count = 0
    for rows in grouped.values():
        gt_ids = {str(row["global_gt_id"]) for row in rows}
        cameras = {row["camera_id"] for row in rows}
        if len(gt_ids) == 1 and len(cameras) > 1:
            count += 1
    return count


def pairwise_unknown_metrics(resolved_rows):
    unknown_rows = [row for row in resolved_rows if row["identity_status"] == "unknown" and row.get("unknown_global_id")]
    tp = fp = fn = tn = 0
    for row_a, row_b in combinations(unknown_rows, 2):
        if row_a["camera_id"] == row_b["camera_id"]:
            continue
        gt_same = str(row_a["global_gt_id"]) == str(row_b["global_gt_id"])
        pred_same = row_a["unknown_global_id"] == row_b["unknown_global_id"]
        if gt_same and pred_same:
            tp += 1
        elif (not gt_same) and pred_same:
            fp += 1
        elif gt_same and (not pred_same):
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "pairwise_tp": tp,
        "pairwise_fp": fp,
        "pairwise_fn": fn,
        "pairwise_tn": tn,
        "pairwise_precision": round(precision, 4),
        "pairwise_recall": round(recall, 4),
        "pairwise_f1": round(f1, 4),
    }


def split_merge_metrics(resolved_rows):
    unknown_by_id = defaultdict(set)
    unknown_by_gt = defaultdict(set)
    for row in resolved_rows:
        if row["identity_status"] != "unknown" or not row.get("unknown_global_id"):
            continue
        unknown_by_id[row["unknown_global_id"]].add(str(row["global_gt_id"]))
        unknown_by_gt[str(row["global_gt_id"])].add(row["unknown_global_id"])
    merge_error_count = sum(1 for gt_ids in unknown_by_id.values() if len(gt_ids) > 1)
    split_gt_count = sum(1 for unknown_ids in unknown_by_gt.values() if len(unknown_ids) > 1)
    return {
        "merge_error_count": merge_error_count,
        "split_gt_count": split_gt_count,
    }


def summarize_variant(name, resolved_rows, decision_logs):
    known_rows = [row for row in resolved_rows if row["identity_status"] == "known"]
    unknown_rows = [row for row in resolved_rows if row["identity_status"] == "unknown"]
    deferred_rows = [row for row in resolved_rows if row["identity_status"] == "deferred"]
    unknown_ids = {row["unknown_global_id"] for row in unknown_rows if row["unknown_global_id"]}
    reused_unknown_ids = {
        unknown_id
        for unknown_id in unknown_ids
        if sum(1 for row in unknown_rows if row["unknown_global_id"] == unknown_id) > 1
    }
    summary = {
        "variant_name": name,
        "total_event_count": len(resolved_rows),
        "known_accept_count": len(known_rows),
        "unknown_event_count": len(unknown_rows),
        "new_unknown_count": sum(1 for log in decision_logs if log["decision"] == "create_new"),
        "unknown_reuse_count": sum(1 for log in decision_logs if log["decision"] == "unknown_reuse"),
        "defer_count": len(deferred_rows),
        "unique_unknown_id_count": len(unknown_ids),
        "reused_unknown_id_count": len(reused_unknown_ids),
        "true_model_based_reuse_count": count_true_model_based_reuse(resolved_rows),
    }
    summary.update(pairwise_unknown_metrics(resolved_rows))
    summary.update(split_merge_metrics(resolved_rows))
    summary.update(summarize_decision_logs(decision_logs))
    return summary


def rank_variants(rows):
    return sorted(
        rows,
        key=lambda row: (
            row["pairwise_f1"],
            row["true_model_based_reuse_count"],
            -row["merge_error_count"],
            -row["split_gt_count"],
            row["unknown_reuse_count"],
        ),
        reverse=True,
    )


def main(config_path: Path):
    config = load_tuning_config(config_path)
    project_root = resolve_path(config_path.parent, config.get("project_root", str(CONFIG_DEFAULT.parents[2])))
    face_demo_config = load_json(resolve_path(project_root, config["face_demo_config"]))
    wildtrack_config = load_json(resolve_path(project_root, config["wildtrack_demo_config"]))
    base_policy, base_policy_runtime = load_association_policy(
        config_path=str(resolve_path(project_root, config["base_policy_config"])),
        base_dir=project_root,
    )
    transition_map, transition_map_runtime = load_camera_transition_map(
        wildtrack_config,
        config_path=str(resolve_path(project_root, config["camera_transition_map_config"])),
        base_dir=project_root,
    )
    topology = build_topology_for_events(wildtrack_config, transition_map)
    output_root = resolve_path(project_root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    feature_cache_path = output_root / "candidate_event_feature_cache.json"
    candidate_events_csv = resolve_path(project_root, config["input"]["candidate_events_csv"])
    candidate_events = parse_candidate_events(read_csv(candidate_events_csv))

    app = FaceAnalysis(
        name=face_demo_config["insightface_runtime"].get("recommended_model_name", "buffalo_l"),
        root=face_demo_config["insightface_runtime"].get("recommended_model_root", "C:/Users/Admin/.insightface"),
        providers=[face_demo_config["insightface_runtime"].get("provider", "CPUExecutionProvider")],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    if feature_cache_path.exists() and not config.get("refresh_feature_cache", False):
        analyzed_events = load_feature_cache(feature_cache_path)
    else:
        analyzed_events = analyze_event_crops(app, candidate_events)
        save_feature_cache(feature_cache_path, analyzed_events)

    manifest_rows, manifest_source = load_manifest_rows(config, face_demo_config, project_root)
    gallery_root = resolve_path(project_root, config["input"].get("known_face_gallery_root", face_demo_config["known_face_gallery_root"]))
    gallery_embeddings_csv = output_root / "known_face_embeddings_eval.csv"
    identity_means, _ = build_gallery_embeddings(app, manifest_rows, project_root, gallery_embeddings_csv)

    variant_rows = []
    decision_log_index_rows = []
    selected_policy = None
    selected_summary = None
    for variant in config["variants"]:
        policy = deep_merge(base_policy, variant.get("overrides", {}))
        resolved_rows, _profiles, _trace_rows, debug_bundle = assign_model_identities(
            analyzed_events,
            identity_means,
            topology,
            unknown_prefix=face_demo_config["unknown_handling"].get("seed_prefix", "UNK"),
            unknown_start=int(face_demo_config["unknown_handling"].get("start_index", 1)),
            policy=policy,
            return_debug_bundle=True,
        )
        summary = summarize_variant(variant["name"], resolved_rows, debug_bundle["decision_logs"])
        variant_rows.append(summary)
        decision_log_index_rows.append(
            {
                "variant_name": variant["name"],
                "decision_log_count": len(debug_bundle["decision_logs"]),
                "resolved_row_count": len(resolved_rows),
            }
        )
        if selected_summary is None:
            selected_policy = policy
            selected_summary = summary

    ranked_rows = rank_variants(variant_rows)
    selected_summary = ranked_rows[0]
    selected_policy = None
    for variant in config["variants"]:
        if variant["name"] == selected_summary["variant_name"]:
            selected_policy = deep_merge(base_policy, variant.get("overrides", {}))
            break

    selected_policy_path = resolve_path(project_root, config["selected_policy_output"])
    selected_policy_path.parent.mkdir(parents=True, exist_ok=True)
    selected_policy_path.write_text(
        yaml.safe_dump({"association_policy": selected_policy}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    write_csv(output_root / "association_policy_sweep.csv", ranked_rows, fieldnames_for_rows(ranked_rows))
    save_json(
        output_root / "association_policy_sweep.json",
        {
            "base_policy_runtime": base_policy_runtime,
            "camera_transition_map_runtime": transition_map_runtime,
            "manifest_source": str(manifest_source),
            "candidate_events_csv": str(candidate_events_csv),
            "ranked_variants": ranked_rows,
            "decision_log_index": decision_log_index_rows,
        },
    )
    save_json(
        output_root / "selected_policy_summary.json",
        {
            "selected_variant": selected_summary["variant_name"],
            "selected_policy_output": str(selected_policy_path),
            "selected_metrics": selected_summary,
            "selection_rule": "max pairwise_f1, then max true_model_based_reuse_count, then min merge_error_count, then min split_gt_count",
        },
    )
    (output_root / "selected_policy_summary.md").write_text(
        "\n".join(
            [
                "# Association Policy Tuning",
                "",
                f"- selected_variant: `{selected_summary['variant_name']}`",
                f"- pairwise_f1: `{selected_summary['pairwise_f1']}`",
                f"- true_model_based_reuse_count: `{selected_summary['true_model_based_reuse_count']}`",
                f"- merge_error_count: `{selected_summary['merge_error_count']}`",
                f"- split_gt_count: `{selected_summary['split_gt_count']}`",
                f"- output_policy: `{selected_policy_path}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"TUNING_OUTPUT_ROOT={output_root}")
    print(f"SELECTED_VARIANT={selected_summary['variant_name']}")
    print(f"PAIRWISE_F1={selected_summary['pairwise_f1']}")
    print(f"TRUE_MODEL_BASED_REUSE_COUNT={selected_summary['true_model_based_reuse_count']}")
    print(f"MERGE_ERROR_COUNT={selected_summary['merge_error_count']}")
    print(f"SPLIT_GT_COUNT={selected_summary['split_gt_count']}")
    print(f"SELECTED_POLICY_OUTPUT={selected_policy_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run association policy tuning from cached candidate events.")
    parser.add_argument("--config", required=True, help="Path to association tuning config YAML.")
    args = parser.parse_args()
    main(Path(args.config).resolve())
