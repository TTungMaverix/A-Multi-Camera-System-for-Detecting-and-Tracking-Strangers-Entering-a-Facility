import argparse
import json
from pathlib import Path

import cv2
import yaml
from insightface.app import FaceAnalysis

from association_core import load_association_policy
from association_core.appearance_evidence import cosine_similarity
from association_core.face_pixel import save_aligned_grayscale_face
from association_core.known_db_runtime import (
    DEFAULT_KNOWN_DB_ROOT,
    DEFAULT_KNOWN_FACE_MANIFEST,
    apply_known_db_defaults,
    ensure_known_db_manifest_rows,
)
from association_core.quality_gate import evaluate_buffered_face_gate
from offline_pipeline.event_builder import get_head_rect, write_image_unicode
from offline_pipeline.orchestrator import run_offline_pipeline
from run_face_resolution_demo import (
    _laplacian_variance,
    build_gallery_embeddings,
    extract_embedding_from_image,
    load_json,
    read_csv,
    resolve_path,
    save_json,
    write_csv,
)
from run_face_resolution_variant import _rewrite_runtime_paths


def _to_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _boolish(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_yaml_payload(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("offline_pipeline", payload)


def _write_runtime_dataset_profile(project_root: Path, output_root: Path):
    source_path = resolve_path(project_root, "insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml")
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    dataset_profile = payload.get("dataset_profile", payload)
    dataset_root = Path(DEFAULT_KNOWN_DB_ROOT).resolve().parent
    dataset_profile["dataset_root"] = str(dataset_root)
    dataset_profile.setdefault("physical_cameras", {})
    dataset_profile["physical_cameras"].setdefault("CAM1", {})
    dataset_profile["physical_cameras"]["CAM1"]["root_dir"] = str((dataset_root / "Camera 1").resolve())
    dataset_profile["physical_cameras"].setdefault("CAM2", {})
    dataset_profile["physical_cameras"]["CAM2"]["root_dir"] = str((dataset_root / "Camera 2").resolve())
    config_root = output_root.parent.parent / "runtime_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "dataset_profile.new_dataset_demo.runtime.yaml"
    config_path.write_text(
        yaml.safe_dump({"dataset_profile": dataset_profile}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def _write_pipeline_config(
    project_root: Path,
    base_config_path: Path,
    output_root: Path,
    pair_id: str,
    detector_conf_threshold: float,
):
    payload = _load_yaml_payload(base_config_path)
    payload["project_root"] = str(project_root)
    payload["output_root"] = str(output_root)
    payload["dataset_profile_config"] = str(_write_runtime_dataset_profile(project_root, output_root))
    payload.setdefault("logical_demo", {})
    payload["logical_demo"]["pair_id"] = pair_id
    payload.setdefault("multi_source_inference", {})
    payload["multi_source_inference"]["conf_threshold"] = float(detector_conf_threshold)
    payload.setdefault("known_gallery", {})
    payload["known_gallery"]["manifest_csv"] = DEFAULT_KNOWN_FACE_MANIFEST
    payload["known_gallery"]["gallery_root"] = DEFAULT_KNOWN_DB_ROOT
    config_root = output_root.parent.parent / "runtime_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / f"{pair_id}_known_face_validation.yaml"
    config_path.write_text(
        yaml.safe_dump({"offline_pipeline": payload}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def _build_debug_runtime_config(base_runtime_config_path: Path, output_root: Path, policy_config: Path, project_root: Path):
    runtime_config = apply_known_db_defaults(load_json(base_runtime_config_path))
    runtime_config = _rewrite_runtime_paths(runtime_config, output_root)
    runtime_config["association_policy_config"] = str(policy_config.resolve())
    runtime_config["known_face_manifest_csv"] = str((project_root / DEFAULT_KNOWN_FACE_MANIFEST).resolve())
    runtime_config["known_face_gallery_root"] = DEFAULT_KNOWN_DB_ROOT
    runtime_config_path = output_root / "runtime" / "face_demo_runtime_config.json"
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_config_path.write_text(json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return runtime_config_path


def _count_entry_events(entry_rows):
    counts = {}
    for row in entry_rows:
        camera_id = row.get("camera_id", "")
        counts[camera_id] = counts.get(camera_id, 0) + 1
    return counts


def _top_known_candidates(face_embedding, identity_means, limit=5):
    rows = []
    if face_embedding is None:
        return rows
    for identity_id, ref_vec in identity_means.items():
        rows.append({"identity_id": identity_id, "score": round(float(cosine_similarity(face_embedding, ref_vec)), 4)})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:limit]


def _load_policy_thresholds(policy_path: Path):
    policy, runtime = load_association_policy(str(policy_path), base_dir=policy_path.parent)
    return policy, runtime


def _probe_camera_faces(project_root: Path, run_output_root: Path, camera_id: str, debug_policy_path: Path, report_root: Path):
    tracks_path = run_output_root / "tracks" / f"{camera_id}_tracks.csv"
    if not tracks_path.exists():
        return {
            "camera_id": camera_id,
            "status": "missing_tracks",
            "track_csv": str(tracks_path),
            "samples": [],
            "known_match_success_count": 0,
            "face_embedding_created_count": 0,
            "best_similarity": 0.0,
        }
    track_rows = read_csv(tracks_path)
    if not track_rows:
        return {
            "camera_id": camera_id,
            "status": "empty_tracks",
            "track_csv": str(tracks_path),
            "samples": [],
            "known_match_success_count": 0,
            "face_embedding_created_count": 0,
            "best_similarity": 0.0,
        }

    policy, _runtime = _load_policy_thresholds(debug_policy_path)
    quality_gate = policy.get("quality_gate", {})
    det_thresh = float(quality_gate.get("face_detector_runtime_threshold", 0.1))
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(Path.home() / ".insightface"),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=det_thresh)

    manifest_path = (project_root / DEFAULT_KNOWN_FACE_MANIFEST).resolve()
    known_root = resolve_path(project_root, DEFAULT_KNOWN_DB_ROOT)
    manifest_rows, manifest_runtime = ensure_known_db_manifest_rows(project_root, manifest_path, known_root)
    embeddings_csv = report_root / "camera2_probe_known_embeddings.csv"
    identity_means, known_gallery_rows = build_gallery_embeddings(app, manifest_rows, project_root, embeddings_csv)

    dataset_profile_path = resolve_path(project_root, "insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml")
    dataset_profile_payload = yaml.safe_load(dataset_profile_path.read_text(encoding="utf-8")) or {}
    dataset_profile = dataset_profile_payload.get("dataset_profile", dataset_profile_payload)
    head_cfg = dataset_profile.get("head_crop", {"top_ratio": 0.02, "bottom_ratio": 0.46, "side_ratio": 0.18})
    known_threshold = 0.75
    threshold_audit = [0.55, 0.65, 0.75]

    sorted_rows = sorted(
        track_rows,
        key=lambda row: (
            _to_int(row.get("area")),
            _to_float(row.get("detection_score")),
            -_to_int(row.get("source_frame_id_actual"), 999999),
        ),
        reverse=True,
    )
    sample_rows = []
    used_keys = set()
    for row in sorted_rows:
        key = (row.get("local_track_id", ""), _to_int(row.get("source_frame_id_actual"), -1))
        if key in used_keys:
            continue
        used_keys.add(key)
        sample_rows.append(row)
        if len(sample_rows) >= 8:
            break

    sample_dir = report_root / "camera2_face_debug_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    best_success = None
    for index, row in enumerate(sample_rows, start=1):
        video_path = Path(row.get("video_path", ""))
        source_frame = _to_int(row.get("source_frame_id_actual"), -1)
        cap = cv2.VideoCapture(str(video_path))
        if source_frame >= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, frame = cap.read()
        cap.release()
        sample = {
            "sample_index": index,
            "camera_id": camera_id,
            "local_track_id": row.get("local_track_id", ""),
            "global_gt_id": row.get("global_gt_id", ""),
            "source_frame_id_actual": source_frame,
            "relative_sec": _to_float(row.get("relative_sec")),
            "detection_score": _to_float(row.get("detection_score")),
            "bbox_area": _to_int(row.get("area")),
            "roi_footpoint_inside": _boolish(row.get("roi_footpoint_inside")),
            "frame_read_ok": bool(ok),
        }
        if not ok or frame is None:
            sample["fail_stage"] = "video_frame_read_failed"
            summary_rows.append(sample)
            continue
        head_rect = get_head_rect(row, head_cfg, frame.shape[1], frame.shape[0])
        x = max(0, _to_int(head_rect.get("x")))
        y = max(0, _to_int(head_rect.get("y")))
        w = max(1, _to_int(head_rect.get("width"), 1))
        h = max(1, _to_int(head_rect.get("height"), 1))
        head_crop = frame[y : y + h, x : x + w]
        head_path = sample_dir / f"{camera_id}_{index:02d}_{row.get('local_track_id','track')}_f{source_frame:04d}_head.png"
        write_image_unicode(head_path, head_crop)
        face_result = extract_embedding_from_image(app, head_path)
        blur_score = _laplacian_variance(head_path)
        face_gate = evaluate_buffered_face_gate(face_result, blur_score, policy=quality_gate)
        gray_path = head_path.with_name(f"{head_path.stem}_aligned_gray.png")
        gray_result = (
            save_aligned_grayscale_face(head_path, gray_path, bbox=face_result.get("bbox"))
            if face_result.get("status") == "ok"
            else {"status": face_result.get("status", "missing"), "path": "", "shape": ""}
        )
        top_matches = _top_known_candidates(face_result.get("embedding"), identity_means, limit=5)
        best_match = top_matches[0] if top_matches else {}
        threshold_hits = {f"known_threshold_{str(value).replace('.', '_')}": bool(best_match and best_match["score"] >= value) for value in threshold_audit}
        sample.update(
            {
                "head_crop_path": str(head_path),
                "face_status": face_result.get("status", ""),
                "face_count": int(face_result.get("face_count", 0) or 0),
                "landmarks_available": bool(face_gate.get("landmarks_available", False)),
                "alignment_status": gray_result.get("status", ""),
                "face_embedding_created": bool(face_result.get("embedding") is not None),
                "face_det_score": round(float(face_result.get("det_score", 0.0) or 0.0), 4),
                "bbox_width": int(face_result.get("bbox_width", 0) or 0),
                "bbox_height": int(face_result.get("bbox_height", 0) or 0),
                "bbox_area_face": int(face_result.get("bbox_area", 0) or 0),
                "blur_score": round(float(blur_score), 4),
                "gate_accept": bool(face_gate.get("accepted_into_buffer", False)),
                "gate_reject_reason": face_gate.get("reject_reason", ""),
                "yaw_deg": round(float(face_gate.get("yaw_deg", 0.0) or 0.0), 4),
                "pitch_deg": round(float(face_gate.get("pitch_deg", 0.0) or 0.0), 4),
                "roll_deg": round(float(face_gate.get("roll_deg", 0.0) or 0.0), 4),
                "top_known_matches": top_matches,
                "best_known_id": best_match.get("identity_id", ""),
                "best_similarity": round(float(best_match.get("score", 0.0) or 0.0), 4),
                "known_accept_threshold": known_threshold,
                "known_match_under_default_threshold": bool(best_match and best_match["score"] >= known_threshold),
                "known_manifest_image_count": int(manifest_runtime.get("image_count", len(manifest_rows))),
                "known_ids_loaded": sorted(identity_means.keys()),
                "embedding_dimension": len(next(iter(identity_means.values()))) if identity_means else 0,
                **threshold_hits,
            }
        )
        if face_result.get("face_count", 0) <= 0:
            sample["fail_stage"] = "face_detector_no_landmark_or_no_face"
        elif face_result.get("embedding") is None:
            sample["fail_stage"] = "face_embedding_not_created"
        elif not top_matches:
            sample["fail_stage"] = "known_db_compare_empty"
        elif not sample["known_match_under_default_threshold"]:
            sample["fail_stage"] = "similarity_below_known_threshold"
        else:
            sample["fail_stage"] = ""
            if best_success is None or sample["best_similarity"] > best_success["best_similarity"]:
                best_success = sample
        summary_rows.append(sample)

    summary_json = report_root / "camera2_face_debug_summary.json"
    save_json(summary_json, summary_rows)
    summary_csv = report_root / "camera2_face_debug_summary.csv"
    write_csv(summary_csv, summary_rows, list(summary_rows[0].keys()) if summary_rows else ["camera_id", "status"])

    return {
        "camera_id": camera_id,
        "status": "ok",
        "track_csv": str(tracks_path),
        "face_debug_summary_json": str(summary_json),
        "face_debug_summary_csv": str(summary_csv),
        "sample_count": len(summary_rows),
        "face_embedding_created_count": sum(1 for row in summary_rows if row.get("face_embedding_created")),
        "face_detected_count": sum(1 for row in summary_rows if int(row.get("face_count", 0) or 0) > 0),
        "landmarks_available_count": sum(1 for row in summary_rows if row.get("landmarks_available")),
        "known_match_success_count": sum(1 for row in summary_rows if row.get("known_match_under_default_threshold")),
        "best_similarity": max([float(row.get("best_similarity", 0.0) or 0.0) for row in summary_rows] or [0.0]),
        "best_success": best_success,
        "known_gallery_rows": len(known_gallery_rows),
        "known_ids_loaded": sorted(identity_means.keys()),
        "detector_threshold": det_thresh,
        "samples": summary_rows,
    }


def _derive_fail_stage(default_summary, debug_summary, probe_summary, entry_counts, camera_id):
    if entry_counts.get(camera_id, 0) <= 0:
        if probe_summary.get("known_match_success_count", 0) > 0:
            return "direction_or_preroll_blocked_before_known_face_runtime"
        return "no_camera2_entry_event_runtime"
    if debug_summary["face_body_usage"]["metrics"].get("face_detected_count", 0) <= 0:
        return "face_detector_no_landmark_or_no_face"
    if debug_summary["face_body_usage"]["metrics"].get("face_embedding_created_count", 0) <= 0:
        return "face_embedding_missing_after_debug_gate"
    if debug_summary["mode_b_true_assoc"].get("known_accept_count", 0) <= 0:
        return "known_similarity_below_threshold"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Run targeted b2 known-face validation and force-extract face debugging.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--base-config",
        default="insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml",
    )
    parser.add_argument("--pair-id", default="b2")
    parser.add_argument("--camera-id", default="C2")
    parser.add_argument(
        "--output-root",
        default="outputs/evaluations/b2_known_face_runtime_validation",
    )
    parser.add_argument(
        "--face-debug-policy-config",
        default="insightface_demo_assets/runtime/config/association_policy.face_force_extract_debug.yaml",
    )
    parser.add_argument("--detector-conf-threshold", type=float, default=0.6)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    validation_root = resolve_path(project_root, args.output_root)
    validation_root.mkdir(parents=True, exist_ok=True)
    base_config_path = resolve_path(project_root, args.base_config)
    default_run_root = validation_root / "offline_runs" / args.pair_id
    debug_run_root = validation_root / "offline_runs" / f"{args.pair_id}_face_force_extract_debug"
    face_debug_policy_path = resolve_path(project_root, args.face_debug_policy_config)

    pipeline_config_path = _write_pipeline_config(
        project_root,
        base_config_path,
        default_run_root,
        args.pair_id,
        detector_conf_threshold=args.detector_conf_threshold,
    )
    run_offline_pipeline(pipeline_config_path)

    base_runtime_config_path = default_run_root / "runtime" / "face_demo_runtime_config.json"
    debug_runtime_config_path = _build_debug_runtime_config(
        base_runtime_config_path,
        debug_run_root,
        face_debug_policy_path,
        project_root,
    )
    from run_face_resolution_demo import main as run_face_resolution_main

    run_face_resolution_main(debug_runtime_config_path)

    default_summary_path = default_run_root / "summaries" / "face_resolution_summary.json"
    debug_summary_path = debug_run_root / "runtime" / "face_resolution_summary.json"
    default_summary = load_json(default_summary_path)
    debug_summary = load_json(debug_summary_path)
    entry_rows = read_csv(default_run_root / "events" / "entry_in_events.csv")
    entry_counts = _count_entry_events(entry_rows)
    probe_summary = _probe_camera_faces(
        project_root,
        default_run_root,
        args.camera_id,
        face_debug_policy_path,
        validation_root / "camera2_face_debug",
    )

    fail_stage = _derive_fail_stage(default_summary, debug_summary, probe_summary, entry_counts, args.camera_id)
    first_c2_row = {}
    c2_tracks = read_csv(default_run_root / "tracks" / f"{args.camera_id}_tracks.csv")
    if c2_tracks:
        first_c2_row = c2_tracks[0]
    c2_event_exists = int(entry_counts.get(args.camera_id, 0)) > 0
    c2_track_exists = len(c2_tracks) > 0
    c2_top_similarity = float(probe_summary.get("best_similarity", 0.0) or 0.0)
    c2_top_k = []
    probe_best = probe_summary.get("best_success") or {}
    if probe_best.get("top_known_matches"):
        c2_top_k = probe_best.get("top_known_matches")
    else:
        for sample in probe_summary.get("samples", []):
            if sample.get("top_known_matches"):
                c2_top_k = sample.get("top_known_matches")
                break
    b2_cam1_path = str((Path(DEFAULT_KNOWN_DB_ROOT).resolve().parent / "Camera 1" / "b2.mp4"))
    b2_cam2_path = str((Path(DEFAULT_KNOWN_DB_ROOT).resolve().parent / "Camera 2" / "b2.mp4"))

    summary = {
        "pair_id": args.pair_id,
        "camera_id": args.camera_id,
        "detector_conf_threshold": float(args.detector_conf_threshold),
        "default_policy_path": str(resolve_path(project_root, "insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml")),
        "face_debug_policy_path": str(face_debug_policy_path),
        "default_run_root": str(default_run_root),
        "default_pipeline_config": str(pipeline_config_path),
        "default_runtime_config": str(base_runtime_config_path),
        "debug_runtime_config": str(debug_runtime_config_path),
        "entry_in_count_per_camera": entry_counts,
        "c2_event_exists": c2_event_exists,
        "c2_track_exists": c2_track_exists,
        "b2_source_files": {
            "camera_1": {"path": b2_cam1_path, "exists": Path(b2_cam1_path).exists()},
            "camera_2": {"path": b2_cam2_path, "exists": Path(b2_cam2_path).exists()},
        },
        "known_db_root": DEFAULT_KNOWN_DB_ROOT,
        "identities_loaded": probe_summary.get("known_ids_loaded", []),
        "known_embedding_count": int(probe_summary.get("known_gallery_rows", 0)),
        "embedding_dimension": int((probe_summary.get("samples", [{}])[0] if probe_summary.get("samples") else {}).get("embedding_dimension", 0)),
        "default_summary": default_summary,
        "debug_summary": debug_summary,
        "camera_probe_summary": probe_summary,
        "face_candidate_count": int(debug_summary["face_body_usage"]["metrics"].get("face_candidate_count", 0)),
        "face_detected_count": int(debug_summary["face_body_usage"]["metrics"].get("face_detected_count", 0)),
        "landmarks_available_count": int(probe_summary.get("landmarks_available_count", 0)),
        "known_match_success_count": int(debug_summary["mode_b_true_assoc"].get("known_accept_count", 0)),
        "debug_face_embedding_created_count": int(
            debug_summary["face_body_usage"]["metrics"].get("face_embedding_created_count", 0)
        ),
        "debug_face_candidate_count": int(debug_summary["face_body_usage"]["metrics"].get("face_candidate_count", 0)),
        "debug_face_detected_count": int(debug_summary["face_body_usage"]["metrics"].get("face_detected_count", 0)),
        "face_embedding_created_count": int(probe_summary.get("face_embedding_created_count", 0)),
        "alignment_status_counts": {
            "ok": sum(1 for row in probe_summary.get("samples", []) if row.get("alignment_status") == "ok"),
            "not_ok": sum(1 for row in probe_summary.get("samples", []) if row.get("alignment_status") != "ok"),
        },
        "top_k_similarity": c2_top_k,
        "top_similarity": round(c2_top_similarity, 4),
        "unknown_created_count": int(debug_summary["mode_b_true_assoc"].get("new_unknown_count", 0)),
        "fail_stage": fail_stage,
        "camera2_first_track_snapshot": {
            "local_track_id": first_c2_row.get("local_track_id", ""),
            "source_frame_id_actual": _to_int(first_c2_row.get("source_frame_id_actual"), -1),
            "relative_sec": _to_float(first_c2_row.get("relative_sec"), 0.0),
            "roi_footpoint_inside": _boolish(first_c2_row.get("roi_footpoint_inside")),
            "detection_score": _to_float(first_c2_row.get("detection_score"), 0.0),
        },
    }
    summary_path = validation_root / "b2_known_face_runtime_validation_summary.json"
    save_json(summary_path, summary)
    report_path = validation_root / "b2_known_face_runtime_validation.md"
    lines = [
        f"# b2 Known Face Runtime Validation",
        "",
        f"- pair_id: `{args.pair_id}`",
        f"- camera_id: `{args.camera_id}`",
        f"- detector_conf_threshold: `{args.detector_conf_threshold}`",
        f"- default_run_root: `{default_run_root}`",
        f"- debug_run_root: `{debug_run_root}`",
        f"- entry_in_count_per_camera: `{json.dumps(entry_counts, ensure_ascii=False)}`",
        f"- c2_event_exists: `{c2_event_exists}`",
        f"- c2_track_exists: `{c2_track_exists}`",
        f"- default_known_accept_count: `{default_summary['mode_b_true_assoc'].get('known_accept_count', 0)}`",
        f"- debug_known_accept_count: `{debug_summary['mode_b_true_assoc'].get('known_accept_count', 0)}`",
        f"- debug_face_embedding_created_count: `{debug_summary['face_body_usage']['metrics'].get('face_embedding_created_count', 0)}`",
        f"- camera_probe_best_similarity: `{probe_summary.get('best_similarity', 0.0)}`",
        f"- fail_stage: `{fail_stage or 'none'}`",
    ]
    best_success = probe_summary.get("best_success") or {}
    if best_success:
        lines.extend(
            [
                "",
                "## Best Camera 2 Probe Match",
                f"- sample_index: `{best_success.get('sample_index', '')}`",
                f"- local_track_id: `{best_success.get('local_track_id', '')}`",
                f"- source_frame_id_actual: `{best_success.get('source_frame_id_actual', '')}`",
                f"- best_known_id: `{best_success.get('best_known_id', '')}`",
                f"- best_similarity: `{best_success.get('best_similarity', '')}`",
                f"- known_match_under_default_threshold: `{best_success.get('known_match_under_default_threshold', False)}`",
                f"- head_crop_path: `{best_success.get('head_crop_path', '')}`",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
