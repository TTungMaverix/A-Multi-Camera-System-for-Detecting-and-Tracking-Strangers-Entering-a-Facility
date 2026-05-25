import argparse
import csv
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np

from association_core.known_db_runtime import (
    DEFAULT_KNOWN_FACE_EMBEDDINGS_CSV,
    DEFAULT_KNOWN_FACE_EMBEDDINGS_PKL,
    load_known_face_embeddings,
)
from calibration_editor import SHAPE_TYPE_PRESETS, build_shape_catalog
from scene_calibration import (
    build_blank_camera_calibration,
    build_runtime_camera_calibration,
    draw_scene_overlay,
    load_scene_calibration,
    probe_frame_from_source,
    save_scene_calibration,
    validate_scene_calibration,
)


DEFAULT_SCENE_CALIBRATION_PATH = "insightface_demo_assets/runtime/config/manual_scene_calibration.wildtrack_4cam_phase.yaml"
DEFAULT_CAMERA_SEQUENCE = ("C1", "C2", "C3", "C4")
KNOWN_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_json_file(path: Path, fallback):
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_file(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if item:
                rows.append(json.loads(item))
    return rows


def load_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _coerce_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_int(value, fallback=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(fallback)


def _parse_camera_segment_sec(value):
    text = str(value or "").strip().lower()
    if text in {"", "auto", "full"}:
        return None
    numeric = float(text)
    if numeric <= 0:
        return None
    return numeric


def artifact_url(path_value: str) -> str:
    if not path_value:
        return ""
    return "/artifact?path=" + quote(path_value, safe="")


def build_browser_event(event):
    payload = dict(event)
    payload["snapshot_url"] = artifact_url(payload.get("snapshot_path", ""))
    payload["head_snapshot_url"] = artifact_url(payload.get("head_snapshot_path", ""))
    return payload


def load_latest_events(output_root: Path):
    latest_events_path = output_root / "events" / "latest_events.json"
    events = load_json_file(latest_events_path, [])
    return [build_browser_event(event) for event in events]


def _timeline_payload_with_urls(payload):
    item = dict(payload)
    item["representative_snapshot_url"] = artifact_url(item.get("representative_snapshot_path", ""))
    item["representative_head_snapshot_url"] = artifact_url(item.get("representative_head_snapshot_path", ""))
    appearances = []
    for appearance in item.get("appearances", []):
        updated = dict(appearance)
        updated["snapshot_url"] = artifact_url(updated.get("best_body_crop", ""))
        updated["head_snapshot_url"] = artifact_url(updated.get("best_head_crop", ""))
        appearances.append(updated)
    item["appearances"] = appearances
    return item


def load_identity_timeline(output_root: Path):
    timeline_json = output_root / "timelines" / "unknown_identity_timeline.json"
    if timeline_json.exists():
        payload = load_json_file(timeline_json, [])
        return [_timeline_payload_with_urls(item) for item in payload]

    latest_events = load_latest_events(output_root)
    grouped = {}
    for event in latest_events:
        identity_id = event.get("identity_id") or event.get("identity_label") or event.get("camera_id")
        grouped.setdefault(
            identity_id,
            {
                "identity_id": identity_id,
                "identity_label": event.get("identity_label") or identity_id,
                "identity_status": event.get("identity_type", ""),
                "appearance_count": 0,
                "camera_sequence": [],
                "first_seen_camera": event.get("camera_id", ""),
                "first_seen_relative_sec": event.get("relative_sec", 0.0),
                "last_seen_camera": event.get("camera_id", ""),
                "last_seen_relative_sec": event.get("relative_sec", 0.0),
                "representative_snapshot_path": event.get("snapshot_path", ""),
                "representative_head_snapshot_path": event.get("head_snapshot_path", ""),
                "appearances": [],
            },
        )
        grouped[identity_id]["appearance_count"] += 1
        grouped[identity_id]["camera_sequence"].append(event.get("camera_id", ""))
        grouped[identity_id]["last_seen_camera"] = event.get("camera_id", "")
        grouped[identity_id]["last_seen_relative_sec"] = event.get("relative_sec", 0.0)
        grouped[identity_id]["appearances"].append(
            {
                "event_id": event.get("event_id", ""),
                "camera_id": event.get("camera_id", ""),
                "relative_sec": event.get("relative_sec", 0.0),
                "relation_type": event.get("direction", ""),
                "zone_id": event.get("zone_id", ""),
                "subzone_id": event.get("subzone_id", ""),
                "identity_status": event.get("identity_type", ""),
                "ui_identity_state": event.get("identity_type", ""),
                "ui_identity_label": event.get("identity_label", ""),
                "modality_primary_used": event.get("modality_primary", ""),
                "modality_secondary_used": event.get("modality_secondary", ""),
                "decision_reason": event.get("decision_reason", ""),
                "reason_code": event.get("reason_code", ""),
                "best_body_crop": event.get("snapshot_path", ""),
                "best_head_crop": event.get("head_snapshot_path", ""),
                "final_score": event.get("final_score", None),
                "body_score": event.get("body_score", None),
                "face_score": event.get("face_score", None),
            }
        )
    return [_timeline_payload_with_urls(item) for item in grouped.values()]


def load_reid_handoffs(output_root: Path):
    edge_summary_path = output_root / "summaries" / "cross_camera_handoff_summary.json"
    edge_summary = load_json_file(edge_summary_path, {})
    edge_count_map = {
        (item.get("src_camera_id", ""), item.get("dst_camera_id", "")): item.get("count", 0)
        for item in edge_summary.get("handoff_edges", [])
    }
    handoffs = []
    for identity in load_identity_timeline(output_root):
        appearances = identity.get("appearances", []) or []
        for index in range(1, len(appearances)):
            previous = appearances[index - 1] or {}
            current = appearances[index] or {}
            from_camera = previous.get("camera_id", "")
            to_camera = current.get("camera_id", "")
            handoffs.append(
                {
                    "global_id": identity.get("identity_label") or identity.get("identity_id") or "UNKNOWN",
                    "identity_id": identity.get("identity_id", ""),
                    "from_camera": from_camera,
                    "to_camera": to_camera,
                    "transition": f"{from_camera} -> {to_camera}",
                    "camera_path": f"{from_camera} -> {to_camera}",
                    "path": f"{from_camera} -> {to_camera}",
                    "observed_delta_sec": max(
                        0.0,
                        float(current.get("relative_sec", 0.0) or 0.0) - float(previous.get("relative_sec", 0.0) or 0.0),
                    ),
                    "method": current.get("modality_primary_used", ""),
                    "reason": current.get("decision_reason") or current.get("reason_code") or "",
                    "score": (
                        current.get("final_score")
                        if current.get("final_score") not in ("", None)
                        else current.get("body_score")
                        if current.get("body_score") not in ("", None)
                        else current.get("face_score")
                    ),
                    "previous_snapshot_url": previous.get("head_snapshot_url") or previous.get("snapshot_url") or "",
                    "current_snapshot_url": current.get("head_snapshot_url") or current.get("snapshot_url") or "",
                    "edge_count": edge_count_map.get((from_camera, to_camera), 0),
                }
            )
    return handoffs


def load_live_summary(output_root: Path):
    live_summary_path = output_root / "summaries" / "live_pipeline_summary.json"
    if live_summary_path.exists():
        return load_json_file(live_summary_path, {})
    face_summary_path = output_root / "summaries" / "face_resolution_summary.json"
    if face_summary_path.exists():
        payload = load_json_file(face_summary_path, {})
        mode_b = payload.get("mode_b_true_assoc", {})
        return {
            "pipeline_name": "offline_replay_summary",
            "live_event_count": mode_b.get("total_event_count", 0),
            "known_event_count": mode_b.get("known_accept_count", 0),
            "unknown_event_count": mode_b.get("unknown_event_count", 0),
            "pending_event_count": mode_b.get("pending_count", 0),
            "body_fallback_used_count": mode_b.get("body_fallback_used_count", 0),
            "avg_latency_sec": 0.0,
            "dropped_frames_total": 0,
            "worker_summaries": {},
        }
    return {}


def load_resolved_event_rows(output_root: Path):
    rows = load_csv_rows(output_root / "events" / "resolved_events.csv")
    return {str(row.get("event_id") or ""): row for row in rows if row.get("event_id")}


def load_association_decisions(output_root: Path):
    return load_jsonl_file(output_root / "association_logs" / "association_decisions.jsonl")


def load_face_resolution_summary(output_root: Path):
    return load_json_file(output_root / "summaries" / "face_resolution_summary.json", {})


def load_face_body_usage_summary(output_root: Path):
    return load_json_file(output_root / "summaries" / "face_body_usage_summary.json", {})


def load_association_summary(output_root: Path):
    return load_json_file(output_root / "association_logs" / "association_summary.json", {})


def load_offline_pipeline_summary(output_root: Path):
    return load_json_file(output_root / "summaries" / "offline_pipeline_summary.json", {})


def load_known_db_summary(project_root: Path):
    embeddings_pkl = resolve_path(project_root, DEFAULT_KNOWN_FACE_EMBEDDINGS_PKL)
    embeddings_csv = resolve_path(project_root, DEFAULT_KNOWN_FACE_EMBEDDINGS_CSV)
    try:
        _gallery, summary = load_known_face_embeddings(
            project_root,
            embeddings_pkl=embeddings_pkl,
            embeddings_csv=embeddings_csv,
        )
        return {
            **summary,
            "source_path": summary.get("source_path", "") or str(embeddings_pkl if embeddings_pkl.exists() else embeddings_csv),
        }
    except Exception as exc:
        return {
            "source_path": "",
            "source_format": "",
            "identity_count": 0,
            "embedding_count": 0,
            "embedding_dimension": 0,
            "sample_first_10_person_ids": [],
            "error": str(exc),
        }


def _probe_video_metadata(video_path):
    metadata = {
        "path": str(video_path or ""),
        "opened": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_sec": 0.0,
    }
    path_text = str(video_path or "")
    if not path_text or not os.path.exists(path_text):
        return metadata
    capture = cv2.VideoCapture(path_text)
    if not capture or not capture.isOpened():
        if capture is not None:
            capture.release()
        return metadata
    metadata["opened"] = True
    metadata["width"] = _coerce_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 0)
    metadata["height"] = _coerce_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)
    metadata["fps"] = _coerce_float(capture.get(cv2.CAP_PROP_FPS), 0.0)
    metadata["frame_count"] = _coerce_int(capture.get(cv2.CAP_PROP_FRAME_COUNT), 0)
    capture.release()
    if metadata["fps"] > 0 and metadata["frame_count"] > 0:
        metadata["duration_sec"] = round(metadata["frame_count"] / metadata["fps"], 3)
    return metadata


def inspect_output_root(output_root: Path):
    required_paths = {
        "latest_events_json": output_root / "events" / "latest_events.json",
        "resolved_events_csv": output_root / "events" / "resolved_events.csv",
        "timeline_json": output_root / "timelines" / "unknown_identity_timeline.json",
        "face_resolution_summary_json": output_root / "summaries" / "face_resolution_summary.json",
        "face_body_usage_summary_json": output_root / "summaries" / "face_body_usage_summary.json",
        "handoff_summary_json": output_root / "summaries" / "cross_camera_handoff_summary.json",
        "association_summary_json": output_root / "association_logs" / "association_summary.json",
        "association_decisions_jsonl": output_root / "association_logs" / "association_decisions.jsonl",
        "offline_pipeline_summary_json": output_root / "summaries" / "offline_pipeline_summary.json",
    }
    present = {key: path.exists() for key, path in required_paths.items()}
    artifacts_found = any(present.values())
    return {
        "output_root": str(output_root),
        "exists": output_root.exists(),
        "artifacts_found": artifacts_found,
        "present": present,
    }


def inspect_calibration_config(scene_calibration_path: Path, demo_state=None):
    status = {
        "path": str(scene_calibration_path),
        "exists": scene_calibration_path.exists(),
        "valid": False,
        "camera_ids": [],
        "errors": [],
        "warnings": [],
        "frame_size_matches": True,
        "frame_size_mismatches": [],
    }
    if not scene_calibration_path.exists():
        status["errors"].append("calibration_config_missing")
        return status
    calibration, runtime_info = load_scene_calibration(
        config_path=str(scene_calibration_path),
        base_dir=scene_calibration_path.parent,
        required=False,
        camera_ids=list(DEFAULT_CAMERA_SEQUENCE),
    )
    cameras = calibration.get("cameras", {}) or {}
    status["camera_ids"] = sorted(cameras.keys())
    status["errors"].extend(list(runtime_info.get("errors", [])))
    status["warnings"].extend(list(runtime_info.get("warnings", [])))
    for camera_id in DEFAULT_CAMERA_SEQUENCE:
        camera_cfg = cameras.get(camera_id, {}) or {}
        role = str(camera_cfg.get("role", "") or "").strip().lower()
        roi = ((camera_cfg.get("processing_roi") or {}).get("polygon") or [])
        entry_line = camera_cfg.get("entry_line", {}) or {}
        if camera_id not in cameras:
            status["errors"].append(f"{camera_id}: missing_camera_config")
            continue
        if role != "entry":
            status["errors"].append(f"{camera_id}: role must be 'entry'")
        if len(roi) < 3:
            status["errors"].append(f"{camera_id}: processing_roi.polygon must contain at least 3 points")
        if len(entry_line.get("points", []) or []) != 2:
            status["errors"].append(f"{camera_id}: entry_line.points must contain exactly 2 points")
        if len(entry_line.get("in_side_point", []) or []) != 2:
            status["errors"].append(f"{camera_id}: entry_line.in_side_point must contain exactly 2 coordinates")
        if not str(camera_cfg.get("anchor_point_mode", "") or "").strip():
            status["errors"].append(f"{camera_id}: anchor_point_mode is required")
        metadata = demo_state.video_metadata_for_camera(camera_id) if demo_state else {}
        if metadata and metadata.get("opened"):
            frame_ref = camera_cfg.get("frame_size_ref", {}) or {}
            width_matches = _coerce_int(frame_ref.get("width"), 0) == _coerce_int(metadata.get("width"), 0)
            height_matches = _coerce_int(frame_ref.get("height"), 0) == _coerce_int(metadata.get("height"), 0)
            if not (width_matches and height_matches):
                status["frame_size_matches"] = False
                status["frame_size_mismatches"].append(
                    {
                        "camera_id": camera_id,
                        "frame_size_ref": {
                            "width": _coerce_int(frame_ref.get("width"), 0),
                            "height": _coerce_int(frame_ref.get("height"), 0),
                        },
                        "video_size": {
                            "width": _coerce_int(metadata.get("width"), 0),
                            "height": _coerce_int(metadata.get("height"), 0),
                        },
                    }
                )
    c1_cfg = cameras.get("C1", {}) or {}
    c2_cfg = cameras.get("C2", {}) or {}
    c3_cfg = cameras.get("C3", {}) or {}
    c4_cfg = cameras.get("C4", {}) or {}
    if c3_cfg and c1_cfg and (
        c3_cfg.get("processing_roi") != c1_cfg.get("processing_roi")
        or c3_cfg.get("entry_line") != c1_cfg.get("entry_line")
    ):
        status["errors"].append("C3 geometry must equal C1 geometry")
    if c4_cfg and c2_cfg and (
        c4_cfg.get("processing_roi") != c2_cfg.get("processing_roi")
        or c4_cfg.get("entry_line") != c2_cfg.get("entry_line")
    ):
        status["errors"].append("C4 geometry must equal C2 geometry")
    status["valid"] = status["exists"] and not status["errors"] and status["frame_size_matches"]
    return status


def _story_tag_for_event(event, resolved_row):
    identity_type = str(event.get("identity_type") or resolved_row.get("identity_status") or "").strip().lower()
    if identity_type == "known":
        return "KNOWN"
    if identity_type == "unknown":
        return "UNKNOWN"
    if identity_type == "pending":
        return "PENDING"
    return identity_type.upper() or "EVENT"


def _build_entry_story(output_root: Path, events, resolved_rows_by_id):
    story = []
    for event in list(events or [])[::-1]:
        resolved_row = resolved_rows_by_id.get(str(event.get("event_id") or ""), {}) or {}
        matched_known_id = str(resolved_row.get("matched_known_id", "") or "").strip()
        matched_known_score = str(resolved_row.get("matched_known_score", "") or "").strip()
        unknown_global_id = str(resolved_row.get("unknown_global_id", "") or event.get("identity_id", "")).strip()
        face_status = str(resolved_row.get("face_embedding_status", "") or "").strip()
        modality_primary = str(
            resolved_row.get("modality_primary_used", "") or event.get("modality_primary", "") or ""
        ).strip()
        note_parts = [
            f"event={event.get('event_id', '')}",
            f"zone={event.get('zone_id', '-')}",
            f"subzone={event.get('subzone_id', '-')}",
        ]
        if matched_known_id:
            score_text = f" @ {matched_known_score}" if matched_known_score not in {"", "0", "0.0"} else ""
            note_parts.append(f"known={matched_known_id}{score_text}")
        elif unknown_global_id:
            note_parts.append(f"unknown={unknown_global_id}")
        if modality_primary:
            note_parts.append(f"modality={modality_primary}")
        if face_status:
            note_parts.append(f"face={face_status}")
        reason_text = resolved_row.get("decision_reason") or resolved_row.get("reason_code") or event.get("decision_reason") or event.get("reason_code") or ""
        if reason_text:
            note_parts.append(f"reason={reason_text}")
        story.append(
            {
                "event_id": event.get("event_id", ""),
                "snapshot_url": event.get("head_snapshot_url") or event.get("snapshot_url") or "",
                "line1": event.get("identity_label") or matched_known_id or unknown_global_id or event.get("identity_id") or "UNKNOWN",
                "line2": f"{event.get('camera_id', '-')} · {str(event.get('direction', 'ENTRY_IN') or 'ENTRY_IN').upper()}",
                "line3": f"t = {_coerce_float(event.get('relative_sec'), 0.0):.1f}s",
                "tag": _story_tag_for_event(event, resolved_row),
                "note": " · ".join(part for part in note_parts if part),
            }
        )
    return story


def _build_reid_story(output_root: Path, handoffs):
    decisions = load_association_decisions(output_root)
    rows = []
    for decision in decisions:
        candidates = list(decision.get("candidate_evaluations", []) or [])
        if not candidates:
            continue
        for candidate in candidates:
            source_camera = str(candidate.get("source_camera_id") or decision.get("source_camera_id") or "").strip()
            target_camera = str(candidate.get("target_camera_id") or decision.get("target_camera_id") or "").strip()
            if not source_camera and not target_camera and not candidate.get("relation_type"):
                continue
            decision_value = str(decision.get("decision") or "").strip().lower()
            accepted = decision_value in {"unknown_reuse", "known_accept"}
            score_value = candidate.get("appearance_primary")
            if score_value in ("", None):
                score_value = candidate.get("body_score")
            if score_value in ("", None):
                score_value = candidate.get("face_score")
            line2_parts = []
            if score_value not in ("", None):
                line2_parts.append(f"score={_coerce_float(score_value, 0.0):.3f}")
            observed_delta = candidate.get("observed_delta_sec") or decision.get("time_delta")
            if observed_delta not in ("", None):
                line2_parts.append(f"Δt={_coerce_float(observed_delta, 0.0):.1f}s")
            relation_type = candidate.get("relation_type") or decision.get("relation_type") or ""
            if relation_type:
                line2_parts.append(f"relation={relation_type}")
            rows.append(
                {
                    "id": candidate.get("candidate_unknown_global_id") or decision.get("gallery_id_before") or "UNKNOWN",
                    "line1": f"{source_camera or '?'} → {target_camera or decision.get('camera_id', '?')}",
                    "line2": " | ".join(line2_parts) if line2_parts else "No score details recorded.",
                    "outcome": "✓ REUSED" if accepted else "✗ NEW UNKNOWN",
                    "color": "green" if accepted else "red",
                    "reason": candidate.get("acceptance_reason")
                    or candidate.get("rejection_reason")
                    or candidate.get("candidate_reason")
                    or decision.get("reason_code")
                    or "",
                    "previous_snapshot_url": "",
                    "current_snapshot_url": "",
                }
            )
    if rows:
        return rows
    fallback_rows = []
    for handoff in handoffs:
        score_value = handoff.get("score")
        delta_text = handoff.get("observed_delta_sec")
        line2 = []
        if score_value not in ("", None):
            line2.append(f"score={_coerce_float(score_value, 0.0):.3f}")
        if delta_text not in ("", None):
            line2.append(f"Δt={_coerce_float(delta_text, 0.0):.1f}s")
        fallback_rows.append(
            {
                "id": handoff.get("global_id") or handoff.get("identity_id") or "UNKNOWN",
                "line1": handoff.get("transition") or handoff.get("camera_path") or handoff.get("path") or "N/A",
                "line2": " | ".join(line2) if line2 else "Observed in timeline only.",
                "outcome": "✓ OBSERVED",
                "color": "green",
                "reason": handoff.get("reason") or handoff.get("method") or "",
                "previous_snapshot_url": handoff.get("previous_snapshot_url", ""),
                "current_snapshot_url": handoff.get("current_snapshot_url", ""),
            }
        )
    return fallback_rows


def _build_identity_journey_story(timeline_rows):
    story = []
    for identity in timeline_rows:
        story.append(
            {
                "id": identity.get("identity_label") or identity.get("identity_id") or "UNKNOWN",
                "cam_path": " → ".join(identity.get("camera_sequence", []) or [])
                or str(identity.get("first_seen_camera") or "-"),
                "first_seen": f"t = {_coerce_float(identity.get('first_seen_relative_sec'), 0.0):.1f}s",
                "last_seen": f"t = {_coerce_float(identity.get('last_seen_relative_sec'), 0.0):.1f}s",
                "count": _coerce_int(identity.get("appearance_count"), 0),
                "snapshot_url": identity.get("representative_head_snapshot_url")
                or identity.get("representative_snapshot_url")
                or "",
                "status": identity.get("identity_status", ""),
            }
        )
    return story


def build_demo_story(output_root: Path, demo_pair_id: str, scene_calibration_path: Path, demo_state=None, known_db_summary=None):
    known_db_summary = dict(known_db_summary or {})
    output_status = inspect_output_root(output_root)
    calibration_status = inspect_calibration_config(scene_calibration_path, demo_state=demo_state)
    warnings = []
    if not calibration_status.get("exists"):
        warnings.append(f"MANUAL_CALIBRATION_REQUIRED_FOR_{str(demo_pair_id or '').upper() or 'PAIR'}")
    elif not calibration_status.get("valid"):
        warnings.append(f"INVALID_CALIBRATION_FOR_{str(demo_pair_id or '').upper() or 'PAIR'}")
    if not output_status.get("artifacts_found"):
        warnings.append("MISSING_DEMO_ARTIFACTS_FOR_PAIR")

    events = load_latest_events(output_root) if output_status.get("artifacts_found") else []
    timeline_rows = load_identity_timeline(output_root) if output_status.get("artifacts_found") else []
    handoffs = load_reid_handoffs(output_root) if output_status.get("artifacts_found") else []
    resolved_rows = load_resolved_event_rows(output_root) if output_status.get("artifacts_found") else {}
    face_resolution_summary = load_face_resolution_summary(output_root) if output_status.get("artifacts_found") else {}
    face_body_usage_summary = load_face_body_usage_summary(output_root) if output_status.get("artifacts_found") else {}
    association_summary = load_association_summary(output_root) if output_status.get("artifacts_found") else {}
    offline_pipeline_summary = load_offline_pipeline_summary(output_root) if output_status.get("artifacts_found") else {}

    face_metrics = face_body_usage_summary.get("metrics", face_body_usage_summary) if isinstance(face_body_usage_summary, dict) else {}
    mode_b = face_resolution_summary.get("mode_b_true_assoc", {}) if isinstance(face_resolution_summary, dict) else {}
    known_db_runtime = face_resolution_summary.get("known_db_runtime", {}) if isinstance(face_resolution_summary, dict) else {}
    association_metrics = association_summary.get("metrics", association_summary) if isinstance(association_summary, dict) else {}
    timings_sec = offline_pipeline_summary.get("timings_sec", {}) if isinstance(offline_pipeline_summary, dict) else {}

    diagnostics = {
        "demo_pair_id": demo_pair_id,
        "output_root": str(output_root),
        "artifacts_found": bool(output_status.get("artifacts_found")),
        "artifacts_present": output_status.get("present", {}),
        "calibration_config_path": str(scene_calibration_path),
        "calibration_valid": bool(calibration_status.get("valid")),
        "calibration_errors": list(calibration_status.get("errors", [])),
        "calibration_warnings": list(calibration_status.get("warnings", [])),
        "known_db_identity_count": _coerce_int(
            known_db_runtime.get("identities_loaded", known_db_summary.get("identity_count", 0)),
            0,
        ),
        "known_db_embedding_count": _coerce_int(
            known_db_runtime.get("embedding_count", known_db_summary.get("embedding_count", 0)),
            0,
        ),
        "known_db_embedding_dimension": known_db_runtime.get(
            "embedding_dimension",
            known_db_summary.get("embedding_dimension", 0),
        ),
        "face_candidate_count": _coerce_int(face_metrics.get("face_candidate_count", "KEY_NOT_FOUND"), 0)
        if "face_candidate_count" in face_metrics
        else "KEY_NOT_FOUND",
        "face_embedding_created_count": _coerce_int(face_metrics.get("face_embedding_created_count", "KEY_NOT_FOUND"), 0)
        if "face_embedding_created_count" in face_metrics
        else "KEY_NOT_FOUND",
        "known_match_success_count": _coerce_int(face_metrics.get("known_face_match_success_count", "KEY_NOT_FOUND"), 0)
        if "known_face_match_success_count" in face_metrics
        else "KEY_NOT_FOUND",
        "unknown_created_count": _coerce_int(mode_b.get("new_unknown_count", association_metrics.get("new_unknown_count", "KEY_NOT_FOUND")), 0)
        if ("new_unknown_count" in mode_b or "new_unknown_count" in association_metrics)
        else "KEY_NOT_FOUND",
        "unknown_reuse_count": _coerce_int(mode_b.get("unknown_reuse_count", association_metrics.get("unknown_reuse_count", "KEY_NOT_FOUND")), 0)
        if ("unknown_reuse_count" in mode_b or "unknown_reuse_count" in association_metrics)
        else "KEY_NOT_FOUND",
        "event_count": len(events),
        "timeline_identity_count": len(timeline_rows),
        "handoff_count": len(handoffs),
        "processing_summary": {
            "known_event_count": mode_b.get("known_event_count", "KEY_NOT_FOUND"),
            "unknown_event_count": mode_b.get("unknown_event_count", "KEY_NOT_FOUND"),
            "pending_count": mode_b.get("pending_count", association_metrics.get("pending_count", "KEY_NOT_FOUND")),
            "body_fallback_used_count": mode_b.get(
                "body_fallback_used_count",
                face_metrics.get("body_fallback_used_count", "KEY_NOT_FOUND"),
            ),
            "face_unusable_event_count": mode_b.get(
                "face_unusable_event_count",
                face_metrics.get("face_unusable_event_count", "KEY_NOT_FOUND"),
            ),
            "total_pipeline_sec": timings_sec.get("total_pipeline_sec", "KEY_NOT_FOUND"),
        },
        "warnings": warnings,
    }

    return {
        "entry_events": _build_entry_story(output_root, events, resolved_rows),
        "reid_evidence": _build_reid_story(output_root, handoffs),
        "identity_journey": _build_identity_journey_story(timeline_rows),
        "system_status": diagnostics,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def resolve_preview_source(project_root: Path, camera_cfg, *, source_type_override="", source_override=""):
    source_type = source_type_override or camera_cfg.get("preview_source_type", "file")
    source_value = source_override or camera_cfg.get("preview_source", "")
    if not source_value:
        raise RuntimeError("preview_source_missing")
    resolved = resolve_path(project_root, source_value)
    return source_type, str(resolved)


def render_calibration_preview(
    calibration,
    camera_id,
    project_root: Path,
    *,
    source_type_override="",
    source_override="",
    frame_idx=0,
    overlay_enabled=True,
):
    camera_cfg = (calibration.get("cameras", {}) or {}).get(camera_id, {})
    if not camera_cfg:
        raise RuntimeError(f"camera_not_found:{camera_id}")
    source_type, source_value = resolve_preview_source(
        project_root,
        camera_cfg,
        source_type_override=source_type_override,
        source_override=source_override,
    )
    frame = probe_frame_from_source(source_type, source_value, frame_idx=int(frame_idx or 0))
    if overlay_enabled:
        runtime_camera = build_runtime_camera_calibration(camera_cfg, frame.shape[1], frame.shape[0])
        frame = draw_scene_overlay(frame, runtime_camera)
    return frame


def read_request_json(handler):
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    data = handler.rfile.read(length)
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


def _json_success(payload):
    return {"ok": True, **payload}


def _camera_folder_for_id(camera_id: str):
    normalized = (camera_id or "").upper()
    if normalized in {"C1", "C3"}:
        return "Camera 1"
    if normalized in {"C2", "C4"}:
        return "Camera 2"
    return None


def _default_camera_description(camera_id: str):
    normalized = (camera_id or "").upper()
    if normalized == "C1":
        return "Logical stream backed by physical Camera 1."
    if normalized == "C2":
        return "Logical stream backed by physical Camera 2."
    if normalized == "C3":
        return "Logical replay of C1."
    if normalized == "C4":
        return "Logical replay of C2."
    return ""


def _candidate_clip_paths(base_path: Path, pair_id: str):
    clip_id = (pair_id or "").strip()
    if not clip_id:
        return []
    direct_paths = [base_path / f"{clip_id}{extension}" for extension in KNOWN_VIDEO_EXTENSIONS]
    nested_paths = [(base_path / clip_id) / f"{clip_id}{extension}" for extension in KNOWN_VIDEO_EXTENSIONS]
    return direct_paths + nested_paths


def _resolve_single_camera_video_source(base_path: Path, pair_id: str, *, camera_folder_name=""):
    for candidate in _candidate_clip_paths(base_path, pair_id):
        if candidate.exists():
            return {"path": candidate.resolve(), "source_note": "EXACT"}
    normalized_pair_id = str(pair_id or "").strip().lower()
    normalized_camera_folder = str(camera_folder_name or base_path.name or "").strip().lower()
    if normalized_pair_id == "d1" and normalized_camera_folder == "camera 1":
        for candidate in _candidate_clip_paths(base_path, "d"):
            if candidate.exists():
                return {"path": candidate.resolve(), "source_note": "ALIAS_D_TO_D1"}
    return {"path": None, "source_note": "MISSING"}


def _resolve_single_camera_video_path(base_path: Path, pair_id: str):
    return _resolve_single_camera_video_source(base_path, pair_id, camera_folder_name=base_path.name).get("path")


def _resolve_clip_video_sources(demo_pair_id: str, dataset_root: str | Path | None) -> dict:
    if not dataset_root or not demo_pair_id:
        return {
            camera_id: {"path": None, "source_note": "MISSING"}
            for camera_id in DEFAULT_CAMERA_SEQUENCE
        }
    dataset_root = Path(dataset_root).resolve()
    camera_1_source = _resolve_single_camera_video_source(
        dataset_root / "Camera 1",
        demo_pair_id,
        camera_folder_name="Camera 1",
    )
    camera_2_source = _resolve_single_camera_video_source(
        dataset_root / "Camera 2",
        demo_pair_id,
        camera_folder_name="Camera 2",
    )
    return {
        "C1": dict(camera_1_source),
        "C2": dict(camera_2_source),
        "C3": dict(camera_1_source),
        "C4": dict(camera_2_source),
    }


def _resolve_clip_video_paths(demo_pair_id: str, dataset_root: str | Path | None) -> dict:
    return {
        camera_id: str(item.get("path")) if item.get("path") else None
        for camera_id, item in _resolve_clip_video_sources(demo_pair_id, dataset_root).items()
    }


def _camera_source_path(dataset_root: Path, pair_id: str, camera_id: str):
    if not dataset_root or not pair_id:
        return None
    camera_folder = _camera_folder_for_id(camera_id)
    if not camera_folder:
        return None
    return _resolve_single_camera_video_source(
        dataset_root / camera_folder,
        pair_id,
        camera_folder_name=camera_folder,
    ).get("path")


def _list_directory_entries(path: Path, limit=40):
    try:
        entries = sorted(item.name for item in path.iterdir())
    except FileNotFoundError:
        return ["<missing directory>"]
    except NotADirectoryError:
        return ["<not a directory>"]
    if len(entries) > limit:
        return entries[:limit] + [f"... ({len(entries) - limit} more entries)"]
    return entries


def _print_video_path_resolution(video_paths: dict, dataset_root: Path | None, source_notes=None):
    source_notes = source_notes or {}
    print("=== VIDEO PATH RESOLUTION ===")
    for camera_id in DEFAULT_CAMERA_SEQUENCE:
        video_path = video_paths.get(camera_id)
        status = "FOUND" if (video_path and os.path.exists(video_path)) else "MISSING"
        source_note = source_notes.get(camera_id, "MISSING")
        if source_note and source_note != "MISSING":
            print(f"  {camera_id}: {video_path} [{status}|{source_note}]")
        else:
            print(f"  {camera_id}: {video_path} [{status}]")
    print("=============================")
    if source_notes.get("C1") == "ALIAS_D_TO_D1":
        print(f"ALIAS_USED: d1 Camera 1 -> {video_paths.get('C1')}")

    missing_camera_1 = not video_paths.get("C1")
    missing_camera_2 = not video_paths.get("C2")
    if not dataset_root or (not missing_camera_1 and not missing_camera_2):
        return

    if missing_camera_1:
        camera_1_dir = Path(dataset_root) / "Camera 1"
        print(f"=== DIRECTORY LISTING: {camera_1_dir} ===")
        for entry in _list_directory_entries(camera_1_dir):
            print(f"  {entry}")
        print("========================================")
    if missing_camera_2:
        camera_2_dir = Path(dataset_root) / "Camera 2"
        print(f"=== DIRECTORY LISTING: {camera_2_dir} ===")
        for entry in _list_directory_entries(camera_2_dir):
            print(f"  {entry}")
        print("========================================")


def _frame_size_from_camera_cfg(camera_cfg):
    size = camera_cfg.get("frame_size_ref") or camera_cfg.get("frame_size") or camera_cfg.get("resolution")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            width = max(1, int(size[0]))
            height = max(1, int(size[1]))
            return width, height
        except (TypeError, ValueError):
            return 1280, 720
    return 1280, 720


def _placeholder_frame(camera_id, phase_label, width=1280, height=720, secondary_text=""):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (16, 11, 8)
    center_y = max(80, height // 2)
    title = f"{camera_id} {phase_label}".strip().upper()
    cv2.putText(frame, title, (40, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (96, 174, 230), 2, cv2.LINE_AA)
    if secondary_text:
        cv2.putText(
            frame,
            secondary_text.upper(),
            (40, min(height - 40, center_y + 52)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (58, 90, 122),
            2,
            cv2.LINE_AA,
        )
    return frame


class VideoStreamer:
    """
    Reads one video file in a background thread and exposes the latest JPEG frame.
    One logical camera owns one streamer instance even if multiple logical cameras
    resolve to the same physical video path.
    """

    _black_jpeg_cache = {}
    _black_cache_lock = threading.Lock()

    def __init__(self, camera_id, video_path, target_fps=15.0, jpeg_quality=80):
        self.camera_id = camera_id
        self.video_path = str(video_path) if video_path else ""
        self.target_fps = max(1.0, float(target_fps or 15.0))
        self.jpeg_quality = max(30, min(95, int(jpeg_quality or 80)))
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._capture = None
        self._current_frame_jpeg = None
        self._last_error = None

    @classmethod
    def get_black_jpeg(cls, width=1280, height=720):
        key = (int(width), int(height))
        with cls._black_cache_lock:
            if key not in cls._black_jpeg_cache:
                frame = np.zeros((key[1], key[0], 3), dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                cls._black_jpeg_cache[key] = encoded.tobytes() if ok else b""
            return cls._black_jpeg_cache[key]

    def is_running(self):
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def _release_capture(self):
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def _open_capture(self):
        self._release_capture()
        if not self.video_path or not os.path.exists(self.video_path):
            self._last_error = f"video_missing:{self.video_path}"
            return False
        capture = cv2.VideoCapture(self.video_path)
        if not capture or not capture.isOpened():
            self._last_error = f"video_open_failed:{self.video_path}"
            if capture is not None:
                capture.release()
            return False
        self._capture = capture
        self._last_error = None
        return True

    def _run(self):
        frame_period = 1.0 / self.target_fps
        if not self._open_capture():
            print(f"[VIDEO_STREAMER] {self.camera_id} open failed: {self._last_error}")
            return

        while not self._stop_event.is_set():
            cycle_started = time.monotonic()
            if self._capture is None and not self._open_capture():
                self._stop_event.wait(0.5)
                continue

            ok, frame = self._capture.read() if self._capture is not None else (False, None)
            if not ok or frame is None:
                if self._capture is not None:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._capture.read()
            if not ok or frame is None:
                self._stop_event.wait(0.1)
                continue

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                with self._lock:
                    self._current_frame_jpeg = encoded.tobytes()

            remaining = frame_period - (time.monotonic() - cycle_started)
            if remaining > 0:
                self._stop_event.wait(remaining)

        self._release_capture()

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run,
                name=f"VideoStreamer-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self):
        with self._lock:
            thread = self._thread
            if thread is None:
                self._release_capture()
                return
            self._stop_event.set()
            self._thread = None
        if thread.is_alive():
            thread.join(timeout=1.5)
        self._release_capture()

    def get_frame(self):
        with self._lock:
            return self._current_frame_jpeg


class DemoPlaybackState:
    def __init__(
        self,
        *,
        presentation_mode="sequential",
        camera_sequence=None,
        camera_segment_sec=12.0,
        gap_seconds=8.0,
        stream_target_fps=15.0,
        dataset_root=None,
        demo_pair_id="",
    ):
        self._lock = threading.Lock()
        self.presentation_mode = presentation_mode or "sequential"
        self.sequence = [item.strip().upper() for item in (camera_sequence or list(DEFAULT_CAMERA_SEQUENCE)) if item.strip()]
        self.camera_segment_sec = None if camera_segment_sec is None else max(1.0, float(camera_segment_sec))
        self.gap_seconds = max(0.0, float(gap_seconds or 0.0))
        self.stream_target_fps = max(1.0, float(stream_target_fps or 15.0))
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None
        self.demo_pair_id = demo_pair_id or ""
        self.demo_start_monotonic = None
        self.video_sources = _resolve_clip_video_sources(self.demo_pair_id, self.dataset_root)
        self.video_paths = {
            camera_id: str(item.get("path")) if item.get("path") else None
            for camera_id, item in self.video_sources.items()
        }
        self.video_source_notes = {
            camera_id: str(item.get("source_note", "MISSING") or "MISSING")
            for camera_id, item in self.video_sources.items()
        }
        self.video_metadata = {
            camera_id: _probe_video_metadata(video_path)
            for camera_id, video_path in self.video_paths.items()
        }
        self.segment_durations_sec = {
            camera_id: self._resolve_segment_duration(camera_id)
            for camera_id in self.sequence
        }
        self._streamers = {
            camera_id: VideoStreamer(camera_id, video_path, target_fps=self.stream_target_fps)
            for camera_id, video_path in self.video_paths.items()
            if video_path
        }
        self._active_stream_camera = None

    def _resolve_segment_duration(self, camera_id):
        if self.camera_segment_sec is not None:
            return float(self.camera_segment_sec)
        metadata = self.video_metadata.get((camera_id or "").upper(), {})
        duration_sec = _coerce_float(metadata.get("duration_sec"), 0.0)
        return max(1.0, duration_sec) if duration_sec > 0 else 12.0

    def _ensure_started(self):
        with self._lock:
            if self.demo_start_monotonic is None:
                self.demo_start_monotonic = time.monotonic()
            return self.demo_start_monotonic

    def _elapsed(self):
        demo_start = self._ensure_started()
        return max(0.0, time.monotonic() - demo_start)

    def current_state(self):
        elapsed = self._elapsed()
        timeline_cursor = 0.0
        for index, camera_id in enumerate(self.sequence):
            segment_start = timeline_cursor
            segment_duration = self.segment_duration_for_camera(camera_id)
            segment_end = segment_start + segment_duration
            if elapsed < segment_end:
                return {
                    "phase": "cam_playing",
                    "active_camera": camera_id,
                    "sequence": list(self.sequence),
                    "phase_start_time": segment_start,
                    "phase_elapsed_sec": elapsed - segment_start,
                    "demo_time_sec": elapsed,
                    "gap_seconds": self.gap_seconds,
                    "camera_segment_sec": segment_duration,
                    "configured_camera_segment_sec": self.camera_segment_sec if self.camera_segment_sec is not None else "auto",
                    "stream_target_fps": self.stream_target_fps,
                    "presentation_mode": self.presentation_mode,
                    "demo_pair_id": self.demo_pair_id,
                }
            timeline_cursor = segment_end
            if index < len(self.sequence) - 1:
                gap_start = timeline_cursor
                gap_end = gap_start + self.gap_seconds
                if elapsed < gap_end:
                    return {
                        "phase": "gap",
                        "active_camera": None,
                        "sequence": list(self.sequence),
                        "phase_start_time": gap_start,
                        "phase_elapsed_sec": elapsed - gap_start,
                        "demo_time_sec": elapsed,
                        "gap_seconds": self.gap_seconds,
                        "camera_segment_sec": segment_duration,
                        "configured_camera_segment_sec": self.camera_segment_sec if self.camera_segment_sec is not None else "auto",
                        "stream_target_fps": self.stream_target_fps,
                        "presentation_mode": self.presentation_mode,
                        "demo_pair_id": self.demo_pair_id,
                    }
                timeline_cursor = gap_end
        return {
            "phase": "done",
            "active_camera": None,
            "sequence": list(self.sequence),
            "phase_start_time": timeline_cursor if self.sequence else None,
            "phase_elapsed_sec": max(0.0, elapsed - timeline_cursor),
            "demo_time_sec": elapsed,
            "gap_seconds": self.gap_seconds,
            "camera_segment_sec": self.segment_duration_for_camera(self.sequence[-1]) if self.sequence else 0.0,
            "configured_camera_segment_sec": self.camera_segment_sec if self.camera_segment_sec is not None else "auto",
            "stream_target_fps": self.stream_target_fps,
            "presentation_mode": self.presentation_mode,
            "demo_pair_id": self.demo_pair_id,
        }

    def frame_index_for_camera(self, camera_id):
        state = self.current_state()
        active_camera = state.get("active_camera")
        if state.get("phase") != "cam_playing" or active_camera != camera_id:
            return None
        phase_elapsed = float(state.get("phase_elapsed_sec", 0.0) or 0.0)
        return max(0, int(round(phase_elapsed * self.stream_target_fps)))

    def preview_source_override(self, camera_id):
        clip_path = self.video_path_for_camera(camera_id)
        if not clip_path:
            return None
        return {"source_type": "file", "source_value": str(clip_path)}

    def video_path_for_camera(self, camera_id):
        return self.video_paths.get((camera_id or "").upper())

    def video_source_note_for_camera(self, camera_id):
        return self.video_source_notes.get((camera_id or "").upper(), "MISSING")

    def video_metadata_for_camera(self, camera_id):
        return dict(self.video_metadata.get((camera_id or "").upper(), {}))

    def segment_duration_for_camera(self, camera_id):
        return float(self.segment_durations_sec.get((camera_id or "").upper(), self.camera_segment_sec or 12.0))

    def streamer_for_camera(self, camera_id):
        return self._streamers.get((camera_id or "").upper())

    def sync_streamers(self, state=None):
        state = state or self.current_state()
        phase = state.get("phase")
        active_camera = state.get("active_camera") if phase == "cam_playing" else None
        for camera_id, streamer in self._streamers.items():
            if camera_id == active_camera:
                streamer.start()
            else:
                streamer.stop()
        with self._lock:
            self._active_stream_camera = active_camera
        return state

    def stop_all_streamers(self):
        for streamer in self._streamers.values():
            streamer.stop()
        with self._lock:
            self._active_stream_camera = None


class LiveDemoRequestHandler(SimpleHTTPRequestHandler):
    server_version = "LiveDemoHTTP/0.4"

    def __init__(
        self,
        *args,
        web_root=None,
        project_root=None,
        output_root=None,
        scene_calibration_path=None,
        demo_state=None,
        known_db_summary=None,
        **kwargs,
    ):
        self.web_root = web_root
        self.project_root = project_root
        self.output_root = output_root
        self.scene_calibration_path = scene_calibration_path
        self.demo_state = demo_state
        self.known_db_summary = dict(known_db_summary or {})
        super().__init__(*args, directory=str(web_root), **kwargs)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path):
        mime_type, _encoding = mimetypes.guess_type(str(file_path))
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_image(self, image, ext=".png"):
        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            return self._send_json({"ok": False, "error": "failed to encode preview image"}, status=500)
        data = encoded.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_jpeg_bytes(self, data, status=200):
        payload = data or VideoStreamer.get_black_jpeg()
        self.send_response(status)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_video_stream_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=mjpegframe")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

    def _load_calibration(self, required=False):
        camera_ids = None
        if not required:
            camera_ids = list(self.demo_state.sequence if self.demo_state else DEFAULT_CAMERA_SEQUENCE)
        calibration, runtime = load_scene_calibration(
            config_path=str(self.scene_calibration_path),
            base_dir=self.project_root,
            required=required,
            camera_ids=camera_ids,
        )
        calibration.setdefault("cameras", {})
        for camera_id in (camera_ids or list((calibration.get("cameras", {}) or {}).keys())):
            camera_cfg = dict((calibration.get("cameras", {}) or {}).get(camera_id, {}) or {})
            metadata = self.demo_state.video_metadata_for_camera(camera_id) if self.demo_state else {}
            camera_cfg.setdefault("camera_id", camera_id)
            if not str(camera_cfg.get("role", "") or "").strip():
                camera_cfg["role"] = "entry"
            if not str(camera_cfg.get("description", "") or "").strip():
                camera_cfg["description"] = _default_camera_description(camera_id)
            if not str(camera_cfg.get("preview_source", "") or "").strip() and self.demo_state:
                camera_cfg["preview_source"] = self.demo_state.video_path_for_camera(camera_id) or ""
            if not str(camera_cfg.get("preview_source_type", "") or "").strip():
                camera_cfg["preview_source_type"] = "file"
            frame_size_ref = dict(camera_cfg.get("frame_size_ref", {}) or {})
            if metadata and metadata.get("opened"):
                width_matches = _coerce_int(frame_size_ref.get("width"), 0) == _coerce_int(metadata.get("width"), 0)
                height_matches = _coerce_int(frame_size_ref.get("height"), 0) == _coerce_int(metadata.get("height"), 0)
                if _coerce_int(frame_size_ref.get("width"), 0) <= 0 or not width_matches:
                    frame_size_ref["width"] = _coerce_int(metadata.get("width"), 0)
                if _coerce_int(frame_size_ref.get("height"), 0) <= 0 or not height_matches:
                    frame_size_ref["height"] = _coerce_int(metadata.get("height"), 0)
            camera_cfg["frame_size_ref"] = frame_size_ref
            calibration["cameras"][camera_id] = camera_cfg
        return calibration, runtime

    def _preview_source_for_camera(self, camera_id, camera_cfg, query):
        query_params = parse_qs(query)
        source_type = query_params.get("source_type", [camera_cfg.get("preview_source_type", "")])[0]
        source_value = query_params.get("source", [camera_cfg.get("preview_source", "")])[0]
        if not source_type or not source_value:
            override = self.demo_state.preview_source_override(camera_id) if self.demo_state else None
            if override:
                source_type = override["source_type"]
                source_value = override["source_value"]
        return resolve_preview_source(
            self.project_root,
            camera_cfg,
            source_type_override=source_type,
            source_override=source_value,
        )

    def _preview_frame(self, calibration, camera_id, query, overlay_enabled=True):
        query_params = parse_qs(query)
        camera_cfg = (calibration.get("cameras", {}) or {}).get(camera_id, {})
        source_type, source_value = self._preview_source_for_camera(camera_id, camera_cfg, query)
        frame_idx = int(query_params.get("frame_idx", ["0"])[0] or 0)
        return render_calibration_preview(
            calibration,
            camera_id,
            self.project_root,
            source_type_override=source_type,
            source_override=source_value,
            frame_idx=frame_idx,
            overlay_enabled=overlay_enabled,
        )

    def _camera_frame(self, calibration, camera_id, query, overlay_enabled=True):
        camera_cfg = (calibration.get("cameras", {}) or {}).get(camera_id, {})
        if not camera_cfg:
            raise RuntimeError(f"camera_not_found:{camera_id}")
        query_params = parse_qs(query)
        explicit_frame_idx = query_params.get("frame_idx", [""])[0]
        if explicit_frame_idx:
            return self._preview_frame(calibration, camera_id, query, overlay_enabled=overlay_enabled)

        playback_state = self.demo_state.current_state() if self.demo_state else {}
        active_camera = playback_state.get("active_camera")
        phase = playback_state.get("phase", "idle")

        if active_camera != camera_id or phase != "cam_playing":
            width, height = _frame_size_from_camera_cfg(camera_cfg)
            phase_text = "TRAVEL GAP" if phase == "gap" else "DONE" if phase == "done" else "STANDBY"
            return _placeholder_frame(camera_id, phase_text, width=width, height=height, secondary_text="inactive camera")

        frame_idx = self.demo_state.frame_index_for_camera(camera_id) if self.demo_state else 0
        source_type, source_value = self._preview_source_for_camera(camera_id, camera_cfg, query)
        return render_calibration_preview(
            calibration,
            camera_id,
            self.project_root,
            source_type_override=source_type,
            source_override=source_value,
            frame_idx=frame_idx or 0,
            overlay_enabled=overlay_enabled,
        )

    def _camera_frame_bytes(self, camera_id):
        if not self.demo_state:
            return VideoStreamer.get_black_jpeg()
        camera_id = (camera_id or "").upper()
        playback_state = self.demo_state.current_state()
        self.demo_state.sync_streamers(playback_state)
        if playback_state.get("phase") != "cam_playing" or playback_state.get("active_camera") != camera_id:
            return VideoStreamer.get_black_jpeg()
        if not self.demo_state.video_path_for_camera(camera_id):
            print(f"[VIDEO_STREAM] missing video path for {camera_id}")
            return VideoStreamer.get_black_jpeg()
        streamer = self.demo_state.streamer_for_camera(camera_id)
        if not streamer:
            return VideoStreamer.get_black_jpeg()
        streamer.start()
        deadline = time.monotonic() + 2.0
        frame = streamer.get_frame()
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
            frame = streamer.get_frame()
        return frame or VideoStreamer.get_black_jpeg()

    def _video_stream_head(self, camera_id):
        camera_id = (camera_id or "").upper()
        if not camera_id:
            return self._send_json({"ok": False, "error": "missing camera_id"}, status=400)
        state = self.demo_state.current_state() if self.demo_state else {}
        if self.demo_state:
            self.demo_state.sync_streamers(state)
        is_active = state.get("phase") == "cam_playing" and state.get("active_camera") == camera_id
        has_video = bool(self.demo_state and self.demo_state.video_path_for_camera(camera_id))
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=mjpegframe" if is_active and has_video else "image/jpeg",
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _serve_video_stream(self, camera_id):
        camera_id = (camera_id or "").upper()
        if not camera_id:
            return self._send_json({"ok": False, "error": "missing camera_id"}, status=400)

        state = self.demo_state.current_state() if self.demo_state else {}
        if self.demo_state:
            self.demo_state.sync_streamers(state)
        is_active = state.get("phase") == "cam_playing" and state.get("active_camera") == camera_id
        video_path = self.demo_state.video_path_for_camera(camera_id) if self.demo_state else None
        if not is_active or not video_path:
            if not video_path:
                print(f"[VIDEO_STREAM] missing video path for {camera_id}")
            return self._send_jpeg_bytes(VideoStreamer.get_black_jpeg())

        streamer = self.demo_state.streamer_for_camera(camera_id) if self.demo_state else None
        if streamer is None:
            print(f"[VIDEO_STREAM] streamer missing for {camera_id}")
            return self._send_jpeg_bytes(VideoStreamer.get_black_jpeg())

        streamer.start()
        deadline = time.monotonic() + 2.0
        frame = streamer.get_frame()
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.05)
            frame = streamer.get_frame()

        self._send_video_stream_headers()
        frame_interval = 1.0 / max(1.0, float(state.get("stream_target_fps", 15.0) or 15.0))
        try:
            while True:
                state = self.demo_state.current_state() if self.demo_state else {}
                if self.demo_state:
                    self.demo_state.sync_streamers(state)
                if state.get("phase") != "cam_playing" or state.get("active_camera") != camera_id:
                    break
                frame = streamer.get_frame() or VideoStreamer.get_black_jpeg()
                header = (
                    b"--mjpegframe\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(header)
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(frame_interval)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        finally:
            if self.demo_state:
                self.demo_state.sync_streamers(self.demo_state.current_state())

    def _camera_config_payload(self):
        calibration, _runtime = self._load_calibration(required=False)
        cameras = []
        for camera_id in (self.demo_state.sequence if self.demo_state else DEFAULT_CAMERA_SEQUENCE):
            camera_cfg = (calibration.get("cameras", {}) or {}).get(camera_id, {})
            override = self.demo_state.preview_source_override(camera_id) if self.demo_state else None
            cameras.append(
                {
                    "camera_id": camera_id,
                    "label": camera_id,
                    "preview_source_type": (override or {}).get("source_type", camera_cfg.get("preview_source_type", "file")),
                    "preview_source": (override or {}).get("source_value", camera_cfg.get("preview_source", "")),
                    "video_path": self.demo_state.video_path_for_camera(camera_id) if self.demo_state else "",
                    "video_source_note": self.demo_state.video_source_note_for_camera(camera_id) if self.demo_state else "MISSING",
                    "video_duration_sec": (
                        (self.demo_state.video_metadata_for_camera(camera_id) or {}).get("duration_sec", 0.0)
                        if self.demo_state
                        else 0.0
                    ),
                    "camera_segment_sec": (
                        self.demo_state.segment_duration_for_camera(camera_id)
                        if self.demo_state
                        else 0.0
                    ),
                    "frame_size_ref": camera_cfg.get("frame_size_ref", []),
                }
            )
        state = self.demo_state.current_state() if self.demo_state else {}
        return {
            "presentation_mode": state.get("presentation_mode", "sequential"),
            "demo_pair_id": state.get("demo_pair_id", ""),
            "sequence": state.get("sequence", list(DEFAULT_CAMERA_SEQUENCE)),
            "camera_segment_sec": state.get("camera_segment_sec", 12.0),
            "gap_seconds": state.get("gap_seconds", 8.0),
            "stream_target_fps": state.get("stream_target_fps", 15.0),
            "output_root": str(self.output_root),
            "scene_calibration_config": str(self.scene_calibration_path),
            "cameras": cameras,
        }

    def _demo_story_payload(self):
        return build_demo_story(
            self.output_root,
            self.demo_state.demo_pair_id if self.demo_state else "",
            Path(self.scene_calibration_path) if self.scene_calibration_path else Path(""),
            demo_state=self.demo_state,
            known_db_summary=self.known_db_summary,
        )

    def _calibration_state_payload(self):
        calibration, runtime = self._load_calibration(required=False)
        errors, warnings = validate_scene_calibration(calibration)
        shape_catalog = {
            camera_id: build_shape_catalog(camera_cfg)
            for camera_id, camera_cfg in ((calibration.get("cameras", {}) or {}).items())
        }
        return _json_success(
            {
                "config_path": str(self.scene_calibration_path),
                "scene_calibration": calibration,
                "runtime": runtime,
                "validation": {"errors": errors, "warnings": warnings},
                "shape_catalog": shape_catalog,
                "shape_presets": SHAPE_TYPE_PRESETS,
            }
        )

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/video-stream":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            return self._video_stream_head(camera_id)
        return super().do_HEAD()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/latest-events":
            return self._send_json({"events": load_latest_events(self.output_root)})
        if parsed.path == "/api/summary":
            return self._send_json(load_live_summary(self.output_root))
        if parsed.path == "/api/timeline":
            return self._send_json({"identities": load_identity_timeline(self.output_root)})
        if parsed.path == "/api/reid-handoffs":
            return self._send_json({"handoffs": load_reid_handoffs(self.output_root)})
        if parsed.path == "/api/demo-story":
            return self._send_json(self._demo_story_payload())
        if parsed.path == "/api/camera-state":
            return self._send_json(self.demo_state.current_state() if self.demo_state else {})
        if parsed.path == "/api/camera-config":
            return self._send_json(self._camera_config_payload())
        if parsed.path == "/api/calibration/state":
            return self._send_json(self._calibration_state_payload())
        if parsed.path == "/api/camera-frame":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            if not camera_id:
                return self._send_json({"ok": False, "error": "missing camera_id"}, status=400)
            return self._send_jpeg_bytes(self._camera_frame_bytes(camera_id))
        if parsed.path == "/api/video-stream":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            return self._serve_video_stream(camera_id)
        if parsed.path == "/api/calibration/preview":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            if not camera_id:
                return self._send_json({"ok": False, "error": "missing camera_id"}, status=400)
            try:
                calibration, _runtime = self._load_calibration(required=False)
                frame = self._preview_frame(
                    calibration,
                    camera_id,
                    parsed.query,
                    overlay_enabled=parse_qs(parsed.query).get("overlay", ["1"])[0] != "0",
                )
            except Exception as exc:
                return self._send_json({"ok": False, "error": str(exc)}, status=400)
            return self._send_image(frame)
        if parsed.path == "/artifact":
            requested = parse_qs(parsed.query).get("path", [""])[0]
            if not requested:
                return self._send_json({"error": "missing path"}, status=400)
            artifact_path = Path(unquote(requested)).resolve()
            if not artifact_path.exists():
                return self._send_json({"error": "artifact not found"}, status=404)
            if not is_within_root(artifact_path, self.project_root):
                return self._send_json({"error": "artifact outside project root"}, status=403)
            return self._send_file(artifact_path)
        if parsed.path in {"/", "/index.html"}:
            return self._send_file(self.web_root / "index.html")
        if parsed.path == "/calibration.html":
            return self._send_file(self.web_root / "calibration.html")
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/calibration/save":
            payload = read_request_json(self)
            scene_calibration = payload.get("scene_calibration", payload)
            if not isinstance(scene_calibration, dict):
                return self._send_json({"ok": False, "error": "invalid calibration payload"}, status=400)
            errors, warnings = validate_scene_calibration(scene_calibration)
            if errors:
                return self._send_json({"ok": False, "error": "invalid calibration", "validation_errors": errors}, status=400)
            save_scene_calibration(self.scene_calibration_path, scene_calibration)
            return self._send_json(_json_success({"warnings": warnings, "config_path": str(self.scene_calibration_path)}))
        if parsed.path == "/api/calibration/reset":
            payload = read_request_json(self)
            camera_id = payload.get("camera_id", "")
            calibration, _runtime = self._load_calibration(required=False)
            if camera_id:
                existing = (calibration.get("cameras", {}) or {}).get(camera_id, {})
                calibration.setdefault("cameras", {})
                calibration["cameras"][camera_id] = build_blank_camera_calibration(camera_id, template=existing)
            else:
                for item_camera_id, existing in list((calibration.get("cameras", {}) or {}).items()):
                    calibration["cameras"][item_camera_id] = build_blank_camera_calibration(item_camera_id, template=existing)
            save_scene_calibration(self.scene_calibration_path, calibration)
            return self._send_json(_json_success({"config_path": str(self.scene_calibration_path)}))
        return self._send_json({"ok": False, "error": "unsupported endpoint"}, status=404)


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the lightweight live event demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-root", default="outputs/live_runs/file_sanity")
    parser.add_argument("--scene-calibration-config", default=DEFAULT_SCENE_CALIBRATION_PATH)
    parser.add_argument("--demo-pair-id", default="")
    parser.add_argument("--presentation-mode", default="sequential")
    parser.add_argument("--camera-sequence", default="C1,C2,C3,C4")
    parser.add_argument(
        "--camera-segment-sec",
        default="12",
        help="Per-camera active duration in seconds. Use 'auto' or 0 for full clip duration.",
    )
    parser.add_argument("--travel-gap-sec", type=float, default=8.0)
    parser.add_argument("--stream-target-fps", type=float, default=15.0)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    dataset_root = resolve_path(project_root, args.dataset_root) if args.dataset_root else None
    output_root = resolve_path(project_root, args.output_root)
    web_root = Path(__file__).resolve().parent / "web_demo"
    scene_calibration_path = resolve_path(project_root, args.scene_calibration_config)
    camera_sequence = [item.strip().upper() for item in args.camera_sequence.split(",") if item.strip()]
    camera_segment_sec = _parse_camera_segment_sec(args.camera_segment_sec)
    demo_state = DemoPlaybackState(
        presentation_mode=args.presentation_mode,
        camera_sequence=camera_sequence or list(DEFAULT_CAMERA_SEQUENCE),
        camera_segment_sec=camera_segment_sec,
        gap_seconds=args.travel_gap_sec,
        stream_target_fps=args.stream_target_fps,
        dataset_root=dataset_root,
        demo_pair_id=args.demo_pair_id,
    )
    known_db_summary = load_known_db_summary(project_root)

    def handler(*handler_args, **handler_kwargs):
        return LiveDemoRequestHandler(
            *handler_args,
            web_root=web_root,
            project_root=project_root,
            output_root=output_root,
            scene_calibration_path=scene_calibration_path,
            demo_state=demo_state,
            known_db_summary=known_db_summary,
            **handler_kwargs,
        )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"LIVE_DEMO_UI=http://{args.host}:{args.port}")
    print(f"LIVE_DEMO_OUTPUT_ROOT={output_root}")
    print(f"SCENE_CALIBRATION_CONFIG={scene_calibration_path}")
    print(f"DEMO_PAIR_ID={args.demo_pair_id}")
    print(f"PRESENTATION_MODE={args.presentation_mode}")
    print(f"CAMERA_SEQUENCE={','.join(camera_sequence or DEFAULT_CAMERA_SEQUENCE)}")
    print(f"CAMERA_SEGMENT_SEC={'auto' if camera_segment_sec is None else camera_segment_sec}")
    print(f"TRAVEL_GAP_SEC={args.travel_gap_sec}")
    print(f"STREAM_TARGET_FPS={args.stream_target_fps}")
    print(
        "KNOWN_DB_RUNTIME="
        + json.dumps(
            {
                "identity_count": known_db_summary.get("identity_count", 0),
                "embedding_count": known_db_summary.get("embedding_count", 0),
                "embedding_dimension": known_db_summary.get("embedding_dimension", 0),
            },
            ensure_ascii=False,
        )
    )
    if dataset_root:
        print(f"DATASET_ROOT={dataset_root}")
    _print_video_path_resolution(demo_state.video_paths, dataset_root, source_notes=demo_state.video_source_notes)
    print("=== CAMERA SEGMENT DURATIONS ===")
    for camera_id in (camera_sequence or DEFAULT_CAMERA_SEQUENCE):
        metadata = demo_state.video_metadata_for_camera(camera_id)
        print(
            f"  {camera_id}: segment_sec={demo_state.segment_duration_for_camera(camera_id)} "
            f"| source_duration_sec={metadata.get('duration_sec', 0.0)}"
        )
    print("===============================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        demo_state.stop_all_streamers()
        server.server_close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
