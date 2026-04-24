import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml

from dataset_profiles import discover_clip_pairs, load_dataset_profile_from_config
from offline_pipeline.orchestrator import load_pipeline_config, resolve_path, run_offline_pipeline
from run_body_tracklet_evaluation import evaluate_run as evaluate_body_tracklet_run


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def copy_if_exists(source_path: Path, destination_path: Path):
    if source_path.exists():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def build_temp_pipeline_config(base_config, pair_id, output_root: Path, project_root: Path):
    config = deepcopy(base_config)
    config["pipeline_name"] = f"new_dataset_eval_{pair_id}"
    config["output_root"] = str(output_root)
    config["project_root"] = str(project_root)
    for key in (
        "dataset_profile_config",
        "face_demo_config",
        "association_policy_config",
        "camera_transition_map_config",
        "scene_calibration_config",
    ):
        value = str(config.get(key, "") or "").strip()
        if value:
            config[key] = str(resolve_path(project_root, value))
    known_gallery = config.get("known_gallery", {}) or {}
    if known_gallery.get("manifest_csv"):
        known_gallery["manifest_csv"] = str(resolve_path(project_root, known_gallery["manifest_csv"]))
    if known_gallery.get("gallery_root"):
        known_gallery["gallery_root"] = str(resolve_path(project_root, known_gallery["gallery_root"]))
    if known_gallery:
        config["known_gallery"] = known_gallery
    config.setdefault("logical_demo", {})
    config["logical_demo"]["pair_id"] = pair_id
    config.setdefault("multi_source_inference", {})
    config["multi_source_inference"].setdefault("cache", {})
    config["multi_source_inference"]["cache"]["enabled"] = False
    config["multi_source_inference"]["cache"]["use_cache"] = False
    config["multi_source_inference"]["cache"]["refresh_cache"] = False
    return {"offline_pipeline": config}


def selected_candidate_from_log(log_row):
    candidates = list(log_row.get("candidate_evaluations", []) or [])
    if not candidates:
        return None
    selected_id = str(log_row.get("selected_candidate_id") or "").strip()
    if selected_id:
        for candidate in candidates:
            if str(candidate.get("candidate_unknown_global_id") or "") == selected_id:
                return candidate
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


def summarize_appearance_vs_topology(decision_logs):
    summary = {
        "decision_count": len(decision_logs),
        "unknown_reuse_count": 0,
        "create_new_unknown_count": 0,
        "appearance_only_pass_count": 0,
        "topology_supported_pass_count": 0,
        "topology_rescued_count": 0,
        "appearance_only_fail_count": 0,
        "records": [],
    }
    for log_row in decision_logs:
        decision = str(log_row.get("decision") or "")
        candidate = selected_candidate_from_log(log_row)
        record = {
            "observation_id": log_row.get("observation_id", ""),
            "camera_id": log_row.get("camera_id", ""),
            "source_camera_id": log_row.get("source_camera_id", ""),
            "target_camera_id": log_row.get("target_camera_id", ""),
            "decision": decision,
            "reason_code": log_row.get("reason_code", ""),
            "appearance_only_body_score": round(float((candidate or {}).get("body_score", 0.0) or 0.0), 4),
            "appearance_primary": round(float((candidate or {}).get("appearance_primary", 0.0) or 0.0), 4),
            "primary_threshold": round(float((log_row.get("thresholds_used", {}) or {}).get("primary_threshold", 0.0) or 0.0), 4),
            "topology_support_level": (candidate or {}).get("topology_support_level", ""),
            "acceptance_reason": (candidate or {}).get("acceptance_reason", ""),
            "relation_type": (candidate or {}).get("relation_type", log_row.get("relation_type", "")),
            "topology_allowed": bool((candidate or {}).get("topology_allowed", False)),
        }
        summary["records"].append(record)
        if decision == "unknown_reuse":
            summary["unknown_reuse_count"] += 1
            if record["acceptance_reason"] == "candidate_threshold_pass":
                summary["appearance_only_pass_count"] += 1
            elif record["acceptance_reason"] == "topology_supported_body_accept":
                summary["topology_supported_pass_count"] += 1
                summary["topology_rescued_count"] += 1
        elif decision == "create_new":
            summary["create_new_unknown_count"] += 1
            if record["reason_code"] == "below_primary_threshold":
                summary["appearance_only_fail_count"] += 1
    return summary


def summarize_face_branch(face_body_summary):
    return {
        "analyzed_event_count": int(face_body_summary.get("analyzed_event_count", 0)),
        "face_detected_count": int(face_body_summary.get("face_detected_count", 0)),
        "face_candidate_count": int(
            face_body_summary.get("face_candidate_count", face_body_summary.get("face_detected_count", 0))
        ),
        "face_best_shot_selected_count": int(
            face_body_summary.get("face_best_shot_selected_count", face_body_summary.get("best_shot_selected_count", 0))
        ),
        "face_embedding_created_count": int(face_body_summary.get("face_embedding_created_count", 0)),
        "face_usable_event_count": int(face_body_summary.get("face_usable_event_count", 0)),
        "face_reject_size_count": int(face_body_summary.get("face_reject_size_count", 0)),
        "face_reject_yaw_count": int(face_body_summary.get("face_reject_yaw_count", 0)),
        "face_reject_pitch_count": int(face_body_summary.get("face_reject_pitch_count", 0)),
        "face_reject_camera_disabled_count": int(face_body_summary.get("face_reject_camera_disabled_count", 0)),
        "face_reject_blur_count": int(face_body_summary.get("face_reject_blur_count", 0)),
        "face_status_counts": dict(face_body_summary.get("face_status_counts", {})),
        "face_unusable_reason_counts": dict(face_body_summary.get("face_unusable_reason_counts", {})),
    }


def per_clip_summary(pair_id, run_root: Path, body_summary, inventory_row=None):
    stage_input_summary = load_json(run_root / "summaries" / "stage_input_summary.json")
    offline_summary = load_json(run_root / "summaries" / "offline_pipeline_summary.json")
    face_resolution_summary = load_json(run_root / "summaries" / "face_resolution_summary.json")
    face_body_summary = load_json(run_root / "summaries" / "face_body_usage_summary.json")
    handoff_summary = load_json(run_root / "summaries" / "cross_camera_handoff_summary.json")
    association_summary = load_json(run_root / "association_logs" / "association_summary.json")
    decision_logs = read_jsonl(run_root / "association_logs" / "association_decisions.jsonl")
    appearance_vs_topology = summarize_appearance_vs_topology(decision_logs)
    face_branch = summarize_face_branch(face_body_summary)
    mode_b = face_resolution_summary.get("mode_b_true_assoc", {})
    return {
        "pair_id": pair_id,
        "run_output_root": str(run_root),
        "inventory": inventory_row or {},
        "stage_input_summary": {
            "source_backend": stage_input_summary.get("source_backend", ""),
            "filtered_track_rows": stage_input_summary.get("filtered_track_rows", 0),
            "per_camera_rows": stage_input_summary.get("per_camera_rows", {}),
        },
        "identity_summary": {
            "total_event_count": mode_b.get("total_event_count", 0),
            "unknown_event_count": mode_b.get("unknown_event_count", 0),
            "unique_unknown_id_count": mode_b.get("unique_unknown_id_count", 0),
            "reused_unknown_id_count": mode_b.get("reused_unknown_id_count", 0),
            "unknown_reuse_count": mode_b.get("unknown_reuse_count", 0),
            "new_unknown_count": mode_b.get("new_unknown_count", 0),
            "true_model_based_reuse_count": mode_b.get("true_model_based_reuse_count", 0),
        },
        "appearance_vs_topology": appearance_vs_topology,
        "face_branch_summary": face_branch,
        "body_tracklet_summary": {
            "comparison_count": body_summary.get("comparison_count", 0),
            "variants": body_summary.get("variants", {}),
            "average_old_single_frame_body_score": body_summary.get("average_old_single_frame_body_score", 0.0),
            "average_best_variant_body_score": body_summary.get("average_best_variant_body_score", 0.0),
        },
        "handoff_summary": handoff_summary,
        "association_summary": association_summary.get("metrics", {}),
        "timings_sec": dict((offline_summary.get("timings_sec", {}) or {})),
    }


def aggregate_overall(per_clip_rows):
    overall = {
        "evaluated_clip_count": len(per_clip_rows),
        "evaluated_clip_ids": [row["pair_id"] for row in per_clip_rows],
        "clips_with_body_comparison_count": 0,
        "appearance_only_pass_count": 0,
        "topology_supported_pass_count": 0,
        "topology_rescued_count": 0,
        "unknown_reuse_count": 0,
        "create_new_unknown_count": 0,
        "face_embedding_created_count": 0,
        "face_best_shot_selected_count": 0,
        "face_candidate_count": 0,
        "multi_camera_identity_clip_count": 0,
        "average_mean_body_score": 0.0,
        "average_quality_aware_body_score": 0.0,
        "average_osnet_x1_0_quality_aware_body_score": 0.0,
    }
    mean_scores = []
    quality_scores = []
    x1_scores = []
    for row in per_clip_rows:
        appearance = row.get("appearance_vs_topology", {})
        face_branch = row.get("face_branch_summary", {})
        handoff_summary = row.get("handoff_summary", {})
        body_summary = row.get("body_tracklet_summary", {}) or {}
        body_variants = body_summary.get("variants", {})
        overall["appearance_only_pass_count"] += int(appearance.get("appearance_only_pass_count", 0))
        overall["topology_supported_pass_count"] += int(appearance.get("topology_supported_pass_count", 0))
        overall["topology_rescued_count"] += int(appearance.get("topology_rescued_count", 0))
        overall["unknown_reuse_count"] += int(appearance.get("unknown_reuse_count", 0))
        overall["create_new_unknown_count"] += int(appearance.get("create_new_unknown_count", 0))
        overall["face_embedding_created_count"] += int(face_branch.get("face_embedding_created_count", 0))
        overall["face_best_shot_selected_count"] += int(face_branch.get("face_best_shot_selected_count", 0))
        overall["face_candidate_count"] += int(face_branch.get("face_candidate_count", 0))
        overall["multi_camera_identity_clip_count"] += 1 if int(handoff_summary.get("multi_camera_identity_count", 0)) > 0 else 0
        if int(body_summary.get("comparison_count", 0)) > 0:
            overall["clips_with_body_comparison_count"] += 1
            mean_variant = (body_variants.get("osnet_mean", {}) or {}).get("average_body_score")
            quality_variant = (body_variants.get("osnet_quality_aware", {}) or {}).get("average_body_score")
            x1_variant = (body_variants.get("osnet_x1_0_quality_aware", {}) or {}).get("average_body_score")
            if mean_variant is not None:
                mean_scores.append(float(mean_variant))
            if quality_variant is not None:
                quality_scores.append(float(quality_variant))
            if x1_variant is not None:
                x1_scores.append(float(x1_variant))
    if mean_scores:
        overall["average_mean_body_score"] = round(sum(mean_scores) / len(mean_scores), 4)
    if quality_scores:
        overall["average_quality_aware_body_score"] = round(sum(quality_scores) / len(quality_scores), 4)
    if x1_scores:
        overall["average_osnet_x1_0_quality_aware_body_score"] = round(sum(x1_scores) / len(x1_scores), 4)
    return overall


def _clip_regression_metrics(clip_row):
    appearance = clip_row.get("appearance_vs_topology", {}) or {}
    records = list(appearance.get("records", []) or [])
    max_body_score = max((float(record.get("appearance_only_body_score", 0.0) or 0.0) for record in records), default=0.0)
    return {
        "total_event_count": int((clip_row.get("identity_summary", {}) or {}).get("total_event_count", 0)),
        "unknown_reuse_count": int((clip_row.get("identity_summary", {}) or {}).get("unknown_reuse_count", 0)),
        "topology_rescued_count": int(appearance.get("topology_rescued_count", 0)),
        "appearance_only_max_body_score": round(float(max_body_score), 4),
    }


def build_regression_summary(per_clip_rows, baseline_output_dir: Path | None = None):
    baseline_output_dir = Path(baseline_output_dir).resolve() if baseline_output_dir else None
    baseline_rows = {}
    if baseline_output_dir and baseline_output_dir.exists():
        baseline_per_clip_dir = baseline_output_dir / "per_clip_evaluation"
        if baseline_per_clip_dir.exists():
            for path in baseline_per_clip_dir.glob("*.json"):
                try:
                    payload = load_json(path)
                except Exception:
                    continue
                baseline_rows[str(payload.get("pair_id", path.stem))] = payload
    per_clip_summary = {}
    for clip_row in per_clip_rows:
        pair_id = clip_row["pair_id"]
        current_metrics = _clip_regression_metrics(clip_row)
        baseline_metrics = _clip_regression_metrics(baseline_rows[pair_id]) if pair_id in baseline_rows else {}
        per_clip_summary[pair_id] = {
            "before": baseline_metrics,
            "after": current_metrics,
            "delta": {
                key: round(float(current_metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0)), 4)
                for key in current_metrics.keys()
            }
            if baseline_metrics
            else {},
        }
    return {
        "baseline_output_dir": str(baseline_output_dir) if baseline_output_dir else "",
        "per_clip": per_clip_summary,
    }


def write_case_notes(output_path: Path, per_clip_rows, inventory_payload=None):
    lines = []
    lines.append("# Qualitative Case Notes")
    lines.append("")
    if inventory_payload:
        summary = inventory_payload.get("summary", {})
        lines.append(f"- total_paired_clips_available: `{summary.get('total_paired_clips_available', 0)}`")
        lines.append(f"- missing_required_clips_for_target_5: `{summary.get('missing_required_clips_for_target_5', 0)}`")
        lines.append("")
    for row in per_clip_rows:
        face_branch = row.get("face_branch_summary", {})
        appearance = row.get("appearance_vs_topology", {})
        body_variants = (row.get("body_tracklet_summary", {}) or {}).get("variants", {})
        mean_score = (body_variants.get("osnet_mean", {}) or {}).get("average_body_score", 0.0)
        quality_score = (body_variants.get("osnet_quality_aware", {}) or {}).get("average_body_score", 0.0)
        x1_score = (body_variants.get("osnet_x1_0_quality_aware", {}) or {}).get("average_body_score", 0.0)
        lines.append(f"## {row['pair_id']}")
        lines.append("")
        lines.append(f"- multi_subject_likely: `{row.get('inventory', {}).get('multi_subject_likely', False)}`")
        lines.append(f"- hard_scenario_likely: `{row.get('inventory', {}).get('hard_scenario_likely', False)}`")
        lines.append(f"- body_comparison_count: `{row.get('body_tracklet_summary', {}).get('comparison_count', 0)}`")
        lines.append(f"- mean_body_score: `{mean_score}`")
        lines.append(f"- quality_aware_body_score: `{quality_score}`")
        lines.append(f"- osnet_x1_0_quality_aware_body_score: `{x1_score}`")
        lines.append(f"- appearance_only_pass_count: `{appearance.get('appearance_only_pass_count', 0)}`")
        lines.append(f"- topology_rescued_count: `{appearance.get('topology_rescued_count', 0)}`")
        lines.append(f"- face_embedding_created_count: `{face_branch.get('face_embedding_created_count', 0)}`")
        lines.append(f"- dominant_face_unusable_reasons: `{face_branch.get('face_unusable_reason_counts', {})}`")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run per-clip New Dataset evaluation across all paired local clips.")
    parser.add_argument(
        "--pipeline-config",
        default="insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml",
    )
    parser.add_argument(
        "--inventory-json",
        default="outputs/evaluations/new_dataset_inventory_phase_current/dataset_inventory.json",
    )
    parser.add_argument(
        "--calibration-reuse-json",
        default="outputs/evaluations/new_dataset_inventory_phase_current/calibration_reuse_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluations/new_dataset_quality_pooling_phase_current",
    )
    parser.add_argument("--baseline-output-dir", default="")
    parser.add_argument("--pair-id", action="append", default=[])
    args = parser.parse_args()

    pipeline_config_path = Path(args.pipeline_config).resolve()
    base_config = load_pipeline_config(pipeline_config_path)
    project_root = resolve_path(pipeline_config_path.parent, base_config.get("project_root", "."))
    _dataset_profile_path, dataset_profile, _runtime = load_dataset_profile_from_config(
        base_config,
        project_root,
        base_dir=pipeline_config_path.parent,
    )
    discovered = discover_clip_pairs(dataset_profile, project_root)
    available_pair_ids = list(discovered.get("available_pair_ids", []))
    requested_pair_ids = list(args.pair_id or []) or available_pair_ids
    pair_ids = [pair_id for pair_id in requested_pair_ids if pair_id in available_pair_ids]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_dir = output_dir / "runtime_configs"
    runtime_config_dir.mkdir(parents=True, exist_ok=True)

    inventory_payload = {}
    inventory_row_by_pair = {}
    inventory_json_path = Path(args.inventory_json).resolve()
    calibration_reuse_json_path = Path(args.calibration_reuse_json).resolve()
    if inventory_json_path.exists():
        inventory_payload = load_json(inventory_json_path)
        for row in inventory_payload.get("pair_rows", []):
            inventory_row_by_pair[row.get("pair_id", "")] = {
                "ready_for_eval": row.get("ready_for_eval", False),
                "multi_subject_likely": row.get("multi_subject_likely", False),
                "hard_scenario_likely": row.get("hard_scenario_likely", False),
                "camera_1_reuse": (row.get("camera_1", {}) or {}).get("calibration_reuse", {}),
                "camera_2_reuse": (row.get("camera_2", {}) or {}).get("calibration_reuse", {}),
            }
        copy_if_exists(inventory_json_path, output_dir / "dataset_inventory.json")
        copy_if_exists(inventory_json_path.with_suffix(".md"), output_dir / "dataset_inventory.md")
    if calibration_reuse_json_path.exists():
        copy_if_exists(calibration_reuse_json_path, output_dir / "calibration_reuse_summary.json")

    per_clip_rows = []
    per_clip_dir = output_dir / "per_clip_evaluation"
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "offline_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for pair_id in pair_ids:
        run_output_root = runs_dir / pair_id
        if run_output_root.exists():
            shutil.rmtree(run_output_root)
        temp_config = build_temp_pipeline_config(base_config, pair_id, run_output_root, project_root)
        temp_config_path = runtime_config_dir / f"{pair_id}.yaml"
        temp_config_path.write_text(yaml.safe_dump(temp_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

        run_offline_pipeline(temp_config_path)
        body_eval_dir = run_output_root / "evaluation" / "body_tracklet"
        body_summary = evaluate_body_tracklet_run(
            run_output_root,
            body_eval_dir,
            str(resolve_path(project_root, base_config.get("association_policy_config", ""))),
        )
        clip_summary = per_clip_summary(
            pair_id,
            run_output_root,
            body_summary,
            inventory_row=inventory_row_by_pair.get(pair_id, {}),
        )
        save_json(per_clip_dir / f"{pair_id}.json", clip_summary)
        per_clip_rows.append(clip_summary)

    overall_summary = aggregate_overall(per_clip_rows)
    appearance_vs_topology_summary = {
        "overall": {
            "appearance_only_pass_count": overall_summary["appearance_only_pass_count"],
            "topology_supported_pass_count": overall_summary["topology_supported_pass_count"],
            "topology_rescued_count": overall_summary["topology_rescued_count"],
            "unknown_reuse_count": overall_summary["unknown_reuse_count"],
            "create_new_unknown_count": overall_summary["create_new_unknown_count"],
        },
        "per_clip": {
            row["pair_id"]: row.get("appearance_vs_topology", {})
            for row in per_clip_rows
        },
    }
    face_branch_summary = {
        "overall": {
            "face_candidate_count": overall_summary["face_candidate_count"],
            "face_best_shot_selected_count": overall_summary["face_best_shot_selected_count"],
            "face_embedding_created_count": overall_summary["face_embedding_created_count"],
        },
        "per_clip": {
            row["pair_id"]: row.get("face_branch_summary", {})
            for row in per_clip_rows
        },
    }

    save_json(
        output_dir / "overall_evaluation_summary.json",
        {
            "inventory_summary": (inventory_payload.get("summary", {}) if inventory_payload else {}),
            "overall": overall_summary,
            "evaluated_pair_ids": pair_ids,
            "per_clip_json_dir": str(per_clip_dir),
        },
    )
    save_json(output_dir / "appearance_vs_topology_summary.json", appearance_vs_topology_summary)
    save_json(output_dir / "face_branch_summary.json", face_branch_summary)
    save_json(
        output_dir / "regression_summary.json",
        build_regression_summary(per_clip_rows, baseline_output_dir=args.baseline_output_dir or None),
    )
    write_case_notes(output_dir / "qualitative_case_notes.md", per_clip_rows, inventory_payload=inventory_payload)

    print(f"EVALUATED_CLIPS={len(per_clip_rows)}")
    print(output_dir / "overall_evaluation_summary.json")


if __name__ == "__main__":
    main()
