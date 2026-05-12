import argparse
import csv
import json
import mimetypes
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2

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


DEFAULT_SCENE_CALIBRATION_PATH = "insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml"


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
            }
        )
    return [_timeline_payload_with_urls(item) for item in grouped.values()]


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


def resolve_preview_source(project_root: Path, camera_cfg, *, source_type_override="", source_override=""):
    source_type = source_type_override or camera_cfg.get("preview_source_type", "file")
    source_value = source_override or camera_cfg.get("preview_source", "")
    if not source_value:
        raise RuntimeError("preview_source_missing")
    resolved = resolve_path(project_root, source_value)
    return source_type, str(resolved)


def resolve_demo_pair_id(output_root: Path, requested_pair_id: str) -> str:
    requested_pair_id = str(requested_pair_id or "").strip()
    if requested_pair_id and requested_pair_id.lower() != "auto":
        return requested_pair_id
    if requested_pair_id.lower() == "auto":
        return output_root.name.strip()
    return ""


def apply_demo_pair_video_sources(calibration, project_root: Path, dataset_root_value: str, pair_id: str):
    """Override camera preview videos so stream panels match the selected clip.

    Scene calibration is source-camera geometry, not per-clip geometry. The
    coordinates stay from calibration, but video streams must use the selected
    pair (a1/a2/a3/b1/...) so the UI does not keep showing a1 while events come
    from another output root.
    """

    if not pair_id:
        return {}
    dataset_root = resolve_path(project_root, dataset_root_value or "New Dataset")
    camera_source_dirs = {
        "C1": "Camera 1",
        "C2": "Camera 2",
        "C3": "Camera 1",
        "C4": "Camera 2",
    }
    applied = {}
    missing = []
    cameras = calibration.get("cameras", {}) or {}
    for camera_id, source_dir in camera_source_dirs.items():
        if camera_id not in cameras:
            continue
        video_path = None
        for suffix in (".mp4", ".avi", ".mov", ".mkv"):
            candidate = dataset_root / source_dir / f"{pair_id}{suffix}"
            if candidate.exists():
                video_path = candidate
                break
        if video_path is None:
            missing.append(str(dataset_root / source_dir / f"{pair_id}.mp4"))
            continue
        cameras[camera_id]["preview_source"] = str(video_path)
        cameras[camera_id]["preview_source_type"] = "file"
        applied[camera_id] = str(video_path)
    if missing:
        raise RuntimeError("demo_pair_video_missing:" + ";".join(missing))
    return applied


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


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class CameraFrameState:
    camera_id: str
    latest_jpeg: bytes = b""
    latest_metadata: dict = field(default_factory=dict)
    frame_index: int = 0
    demo_time_sec: float = 0.0
    updated_at: float = 0.0
    fps_estimate: float = 0.0
    dropped_frame_count: int = 0
    error: str = ""


class FrameBufferHub:
    def __init__(self):
        self._lock = threading.RLock()
        self._states = {}

    def update(self, camera_id, jpeg_bytes, metadata):
        now = time.time()
        with self._lock:
            state = self._states.setdefault(camera_id, CameraFrameState(camera_id=camera_id))
            previous_update = state.updated_at
            state.latest_jpeg = jpeg_bytes
            state.latest_metadata = dict(metadata or {})
            state.frame_index = int(state.latest_metadata.get("frame_index", state.frame_index + 1))
            state.demo_time_sec = float(state.latest_metadata.get("demo_time_sec", 0.0) or 0.0)
            state.updated_at = now
            if previous_update > 0 and now > previous_update:
                instant_fps = 1.0 / max(now - previous_update, 1e-6)
                state.fps_estimate = round((state.fps_estimate * 0.85) + (instant_fps * 0.15), 2) if state.fps_estimate else round(instant_fps, 2)
            state.error = ""

    def set_error(self, camera_id, error):
        with self._lock:
            state = self._states.setdefault(camera_id, CameraFrameState(camera_id=camera_id))
            state.error = str(error)
            state.updated_at = time.time()

    def get(self, camera_id):
        with self._lock:
            state = self._states.get(camera_id)
            if not state:
                return None
            return CameraFrameState(
                camera_id=state.camera_id,
                latest_jpeg=state.latest_jpeg,
                latest_metadata=dict(state.latest_metadata),
                frame_index=state.frame_index,
                demo_time_sec=state.demo_time_sec,
                updated_at=state.updated_at,
                fps_estimate=state.fps_estimate,
                dropped_frame_count=state.dropped_frame_count,
                error=state.error,
            )

    def snapshot(self):
        with self._lock:
            return {
                camera_id: {
                    "camera_id": state.camera_id,
                    "frame_index": state.frame_index,
                    "demo_time_sec": round(state.demo_time_sec, 3),
                    "updated_at": state.updated_at,
                    "latest_frame_age_ms": round((time.time() - state.updated_at) * 1000.0, 1) if state.updated_at else None,
                    "fps_estimate": state.fps_estimate,
                    "dropped_frame_count": state.dropped_frame_count,
                    "error": state.error,
                    **state.latest_metadata,
                }
                for camera_id, state in sorted(self._states.items())
            }


class SequentialDemoClock:
    """Presentation clock for non-overlap demo playback.

    The thesis demo uses C3/C4 as delayed logical replays. Showing all four
    streams at once makes the non-overlap travel-time story hard to follow, so
    this clock exposes one active camera segment at a time with an explicit
    travel gap before the next camera becomes active.
    """

    def __init__(self, camera_sequence, segment_sec=10.0, travel_gap_sec=3.0):
        self.camera_sequence = [str(item).strip() for item in camera_sequence if str(item).strip()]
        self.segment_sec = max(1.0, float(segment_sec or 10.0))
        self.travel_gap_sec = max(0.0, float(travel_gap_sec or 0.0))
        self.started_at = time.time()

    def phase(self):
        if not self.camera_sequence:
            return {
                "presentation_mode": "simultaneous",
                "phase_kind": "active",
                "active_camera": "",
                "phase_remaining_sec": 0.0,
            }
        slot_sec = self.segment_sec + self.travel_gap_sec
        cycle_sec = max(slot_sec * len(self.camera_sequence), 1.0)
        elapsed = (time.time() - self.started_at) % cycle_sec
        slot_index = min(int(elapsed // slot_sec), len(self.camera_sequence) - 1)
        slot_pos = elapsed - (slot_index * slot_sec)
        current_camera = self.camera_sequence[slot_index]
        next_camera = self.camera_sequence[(slot_index + 1) % len(self.camera_sequence)]
        if slot_pos < self.segment_sec:
            return {
                "presentation_mode": "sequential",
                "phase_kind": "active",
                "active_camera": current_camera,
                "next_camera": next_camera,
                "travel_from_camera": "",
                "travel_to_camera": "",
                "phase_elapsed_sec": round(slot_pos, 3),
                "phase_remaining_sec": round(self.segment_sec - slot_pos, 3),
                "segment_sec": self.segment_sec,
                "travel_gap_sec": self.travel_gap_sec,
                "camera_sequence": self.camera_sequence,
            }
        travel_elapsed = slot_pos - self.segment_sec
        return {
            "presentation_mode": "sequential",
            "phase_kind": "travel_gap",
            "active_camera": "",
            "next_camera": next_camera,
            "travel_from_camera": current_camera,
            "travel_to_camera": next_camera,
            "phase_elapsed_sec": round(travel_elapsed, 3),
            "phase_remaining_sec": round(self.travel_gap_sec - travel_elapsed, 3),
            "segment_sec": self.segment_sec,
            "travel_gap_sec": self.travel_gap_sec,
            "camera_sequence": self.camera_sequence,
        }


def _draw_text_panel(frame, lines, origin=(14, 26)):
    for index, line in enumerate(lines):
        y = origin[1] + (index * 25)
        cv2.putText(frame, str(line), (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 4, cv2.LINE_AA)
        cv2.putText(frame, str(line), (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)


def _row_bbox(row):
    return (
        int(float(row.get("xmin", row.get("bbox_xmin", 0)) or 0)),
        int(float(row.get("ymin", row.get("bbox_ymin", 0)) or 0)),
        int(float(row.get("xmax", row.get("bbox_xmax", 0)) or 0)),
        int(float(row.get("ymax", row.get("bbox_ymax", 0)) or 0)),
    )


class OfflineReplayFrameWorker(threading.Thread):
    def __init__(
        self,
        camera_id,
        camera_cfg,
        runtime_camera,
        project_root: Path,
        output_root: Path,
        hub: FrameBufferHub,
        target_fps=15.0,
        demo_clock=None,
    ):
        super().__init__(name=f"frame-worker-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.camera_cfg = camera_cfg
        self.runtime_camera = runtime_camera
        self.project_root = project_root
        self.output_root = output_root
        self.hub = hub
        self.target_fps = max(1.0, float(target_fps or 15.0))
        self.demo_clock = demo_clock
        self.stop_event = threading.Event()
        self.track_rows_by_source_frame = defaultdict(list)
        self.event_rows_by_source_frame = defaultdict(list)
        self._load_artifact_rows()

    def _load_artifact_rows(self):
        for row in read_csv_rows(self.output_root / "tracks" / f"{self.camera_id}_tracks.csv"):
            frame_id = int(float(row.get("source_frame_id_actual", row.get("frame_id", 0)) or 0))
            self.track_rows_by_source_frame[frame_id].append(row)
        for row in read_csv_rows(self.output_root / "events" / "entry_in_events.csv"):
            if row.get("camera_id") != self.camera_id:
                continue
            frame_id = int(float(row.get("source_frame_idx", row.get("frame_id", 0)) or 0))
            self.event_rows_by_source_frame[frame_id].append(row)

    def stop(self):
        self.stop_event.set()

    def _annotate(self, frame, source_frame_idx, fps, phase=None):
        phase = phase or {}
        frame = draw_scene_overlay(frame, self.runtime_camera)
        active_tracks = []
        for row in self.track_rows_by_source_frame.get(source_frame_idx, []):
            x1, y1, x2, y2 = _row_bbox(row)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 2)
            local_track_id = str(row.get("local_track_id", row.get("global_gt_id", "")) or "")
            label = f"{self.camera_id} T{local_track_id}"
            cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 220, 80), 2, cv2.LINE_AA)
            active_tracks.append({"local_track_id": local_track_id, "bbox": [x1, y1, x2, y2]})
        event_rows = self.event_rows_by_source_frame.get(source_frame_idx, [])
        for row in event_rows:
            _draw_text_panel(
                frame,
                [
                    f"ENTRY_IN {row.get('event_id', '')}",
                    f"mode={row.get('direction_accept_mode', '')} reason={row.get('direction_reason', '')[:64]}",
                ],
                origin=(16, 86),
            )
        demo_time_sec = source_frame_idx / max(fps, 1e-6)
        _draw_text_panel(
            frame,
            [
                f"{self.camera_id} OFFLINE_REPLAY_STREAM",
                f"source_frame={source_frame_idx} demo_time={demo_time_sec:.2f}s",
                f"tracks={len(active_tracks)} entry_events={len(event_rows)}",
            ],
        )
        return frame, {
            "camera_id": self.camera_id,
            "mode": "OFFLINE_REPLAY_STREAM",
            "frame_index": source_frame_idx,
            "demo_time_sec": round(demo_time_sec, 3),
            "active_tracks": active_tracks,
            "entry_events": event_rows,
            "event_count": len(event_rows),
            "track_count": len(active_tracks),
            "video_source": self.camera_cfg.get("preview_source", ""),
            "demo_pair_id": Path(str(self.camera_cfg.get("preview_source", ""))).stem,
            "presentation_mode": phase.get("presentation_mode", "simultaneous"),
            "phase_kind": phase.get("phase_kind", "active"),
            "is_active_camera": phase.get("presentation_mode") != "sequential" or phase.get("active_camera") == self.camera_id,
            "active_camera": phase.get("active_camera", self.camera_id),
            "next_camera": phase.get("next_camera", ""),
            "travel_from_camera": phase.get("travel_from_camera", ""),
            "travel_to_camera": phase.get("travel_to_camera", ""),
            "phase_remaining_sec": phase.get("phase_remaining_sec", 0.0),
            "camera_sequence": phase.get("camera_sequence", []),
        }

    def _standby(self, frame, phase):
        frame = draw_scene_overlay(frame.copy(), self.runtime_camera)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.48, frame, 0.52, 0)
        if phase.get("phase_kind") == "travel_gap":
            status = f"FAKE TRAVEL TIME: {phase.get('travel_from_camera')} -> {phase.get('travel_to_camera')}"
        else:
            status = f"WAITING FOR TURN. Active camera: {phase.get('active_camera')}"
        _draw_text_panel(
            frame,
            [
                f"{self.camera_id} STANDBY",
                status,
                f"next={phase.get('next_camera', '-')} remaining={phase.get('phase_remaining_sec', 0):.1f}s",
            ],
            origin=(18, 34),
        )
        return frame, {
            "camera_id": self.camera_id,
            "mode": "SEQUENTIAL_STANDBY",
            "frame_index": 0,
            "demo_time_sec": 0.0,
            "active_tracks": [],
            "entry_events": [],
            "event_count": 0,
            "track_count": 0,
            "video_source": self.camera_cfg.get("preview_source", ""),
            "demo_pair_id": Path(str(self.camera_cfg.get("preview_source", ""))).stem,
            "presentation_mode": phase.get("presentation_mode", "sequential"),
            "phase_kind": phase.get("phase_kind", "travel_gap"),
            "is_active_camera": False,
            "active_camera": phase.get("active_camera", ""),
            "next_camera": phase.get("next_camera", ""),
            "travel_from_camera": phase.get("travel_from_camera", ""),
            "travel_to_camera": phase.get("travel_to_camera", ""),
            "phase_remaining_sec": phase.get("phase_remaining_sec", 0.0),
            "camera_sequence": phase.get("camera_sequence", []),
        }

    def run(self):
        try:
            source_type, source_value = resolve_preview_source(self.project_root, self.camera_cfg)
            if source_type != "file":
                raise RuntimeError(f"unsupported_stream_source_type:{source_type}")
            capture = cv2.VideoCapture(source_value)
            if not capture.isOpened():
                raise RuntimeError(f"failed_to_open_video:{source_value}")
            fps = capture.get(cv2.CAP_PROP_FPS) or self.target_fps
            ok, first_frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed_to_read_first_frame:{source_value}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_interval = 1.0 / self.target_fps
            source_frame_idx = 0
            was_active = False
            try:
                while not self.stop_event.is_set():
                    loop_started = time.time()
                    phase = self.demo_clock.phase() if self.demo_clock else {"presentation_mode": "simultaneous", "phase_kind": "active", "active_camera": self.camera_id}
                    is_active = phase.get("presentation_mode") != "sequential" or phase.get("active_camera") == self.camera_id
                    if not is_active:
                        standby_frame, metadata = self._standby(first_frame, phase)
                        ok, encoded = cv2.imencode(".jpg", standby_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                        if ok:
                            self.hub.update(self.camera_id, encoded.tobytes(), metadata)
                        was_active = False
                        time.sleep(min(0.5, frame_interval))
                        continue
                    if not was_active:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        source_frame_idx = 0
                        was_active = True
                    ok, frame = capture.read()
                    if not ok:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        source_frame_idx = 0
                        continue
                    annotated, metadata = self._annotate(frame, source_frame_idx, fps, phase=phase)
                    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if ok:
                        self.hub.update(self.camera_id, encoded.tobytes(), metadata)
                    source_frame_idx += 1
                    elapsed = time.time() - loop_started
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
            finally:
                capture.release()
        except Exception as exc:
            self.hub.set_error(self.camera_id, exc)


def load_reid_handoffs(output_root: Path):
    summary_path = output_root / "summaries" / "cross_camera_handoff_summary.json"
    payload = load_json_file(summary_path, {})
    decisions = load_recent_association_decisions(output_root, limit=200)
    decision_handoffs = []
    for row in decisions:
        if row.get("decision") != "unknown_reuse" and row.get("reason_code") != "unknown_reuse":
            continue
        source_camera = row.get("source_camera_id") or row.get("from_camera") or ""
        target_camera = row.get("target_camera_id") or row.get("to_camera") or row.get("camera_id") or ""
        if not source_camera or not target_camera or source_camera == target_camera:
            continue
        thresholds = row.get("thresholds_used", {}) or {}
        decision_handoffs.append(
            {
                "unknown_global_id": row.get("gallery_id_after") or row.get("selected_candidate_id") or "",
                "source_camera_id": source_camera,
                "target_camera_id": target_camera,
                "transition_edge": f"{source_camera} -> {target_camera}",
                "reason_code": row.get("reason_code", ""),
                "acceptance_reason": (row.get("candidate_evaluations") or [{}])[0].get("acceptance_reason", ""),
                "observed_delta_sec": row.get("time_delta", ""),
                "FaceScore": row.get("FaceScore", row.get("face_score", "N/A")),
                "BodyScore": row.get("BodyScore", row.get("body_score", "N/A")),
                "TimeScore": row.get("TimeScore", row.get("time_score", "N/A")),
                "TopologyScore": row.get("TopologyScore", row.get("topology_score", "N/A")),
                "final_score": row.get("final_score", row.get("final_total_score", "N/A")),
                "score_threshold": thresholds.get("decision_score_threshold", row.get("score_threshold", "N/A")),
                "score_formula": row.get("score_formula", "Score = a*FaceScore + b*BodyScore + c*TimeScore + d*TopologyScore"),
            }
        )
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["decision_handoffs"] = decision_handoffs
        return payload
    return {"handoffs": payload, "decision_handoffs": decision_handoffs}


def load_recent_association_decisions(output_root: Path, limit=50):
    path = output_root / "association_logs" / "association_decisions.jsonl"
    rows = deque(maxlen=max(1, int(limit or 50)))
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(rows)


class LiveDemoRequestHandler(SimpleHTTPRequestHandler):
    server_version = "LiveDemoHTTP/0.2"

    def __init__(self, *args, web_root=None, project_root=None, output_root=None, scene_calibration_path=None, frame_hub=None, **kwargs):
        self.web_root = web_root
        self.project_root = project_root
        self.output_root = output_root
        self.scene_calibration_path = scene_calibration_path
        self.frame_hub = frame_hub or FrameBufferHub()
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

    def _send_jpeg(self, data: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
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

    def _load_calibration(self, required=False):
        calibration, runtime = load_scene_calibration(
            config_path=str(self.scene_calibration_path),
            base_dir=self.project_root,
            required=required,
        )
        return calibration, runtime

    def _preview_source_for_camera(self, camera_cfg, query):
        source_type = parse_qs(query).get("source_type", [camera_cfg.get("preview_source_type", "file")])[0]
        source_value = parse_qs(query).get("source", [camera_cfg.get("preview_source", "")])[0]
        return resolve_preview_source(
            self.project_root,
            camera_cfg,
            source_type_override=source_type,
            source_override=source_value,
        )

    def _preview_frame(self, calibration, camera_id, query, overlay_enabled=True):
        query_params = parse_qs(query)
        camera_cfg = (calibration.get("cameras", {}) or {}).get(camera_id, {})
        source_type, source_value = self._preview_source_for_camera(camera_cfg, query)
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

    def _camera_config_payload(self):
        calibration, runtime = self._load_calibration(required=False)
        cameras = {}
        for camera_id, cfg in ((calibration.get("cameras", {}) or {}).items()):
            cameras[camera_id] = {
                "camera_id": camera_id,
                "role": cfg.get("role", ""),
                "description": cfg.get("description", ""),
                "preview_source": cfg.get("preview_source", ""),
                "preview_source_type": cfg.get("preview_source_type", ""),
                "anchor_point_mode": cfg.get("anchor_point_mode", ""),
                "logical_replay_note": "C3/C4 are logical replay streams; API must not fabricate ENTRY_IN events.",
            }
        return {
            "project_root": str(self.project_root),
            "output_root": str(self.output_root),
            "scene_calibration_path": str(self.scene_calibration_path),
            "runtime": runtime,
            "cameras": cameras,
        }

    def _camera_state_payload(self, camera_id=""):
        state = self.frame_hub.snapshot()
        latest_events = load_latest_events(self.output_root)
        latest_by_camera = defaultdict(list)
        for event in latest_events:
            latest_by_camera[event.get("camera_id", "")].append(event)
        if camera_id:
            payload = state.get(camera_id, {"camera_id": camera_id, "error": "camera_state_not_ready"})
            payload["latest_events"] = latest_by_camera.get(camera_id, [])
            return {"camera": payload}
        for item_camera_id, payload in state.items():
            payload["latest_events"] = latest_by_camera.get(item_camera_id, [])
        return {"mode": "OFFLINE_REPLAY_STREAM", "cameras": state}

    def _stream_camera(self, camera_id, interval_sec=0.066):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last_frame_index = None
        while True:
            state = self.frame_hub.get(camera_id)
            if state is None or not state.latest_jpeg:
                time.sleep(interval_sec)
                continue
            if last_frame_index == state.frame_index:
                time.sleep(interval_sec)
                continue
            last_frame_index = state.frame_index
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(state.latest_jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(state.latest_jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                break
            time.sleep(interval_sec)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/latest-events":
            return self._send_json({"events": load_latest_events(self.output_root)})
        if parsed.path == "/api/summary":
            return self._send_json(load_live_summary(self.output_root))
        if parsed.path == "/api/timeline":
            return self._send_json({"identities": load_identity_timeline(self.output_root)})
        if parsed.path == "/api/reid-handoffs":
            return self._send_json(load_reid_handoffs(self.output_root))
        if parsed.path == "/api/association-decisions":
            limit = int(parse_qs(parsed.query).get("limit", ["50"])[0] or 50)
            return self._send_json({"decisions": load_recent_association_decisions(self.output_root, limit=limit)})
        if parsed.path == "/api/camera-config":
            return self._send_json(self._camera_config_payload())
        if parsed.path == "/api/camera-state":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            return self._send_json(self._camera_state_payload(camera_id=camera_id))
        if parsed.path == "/api/camera-frame":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            if not camera_id:
                return self._send_json({"error": "missing camera_id"}, status=400)
            state = self.frame_hub.get(camera_id)
            if state is None or not state.latest_jpeg:
                return self._send_json({"error": "camera_frame_not_ready", "camera_id": camera_id}, status=503)
            return self._send_jpeg(state.latest_jpeg)
        if parsed.path == "/api/camera-stream":
            camera_id = parse_qs(parsed.query).get("camera_id", [""])[0]
            if not camera_id:
                return self._send_json({"error": "missing camera_id"}, status=400)
            return self._stream_camera(camera_id)
        if parsed.path.startswith("/stream/camera/") and parsed.path.endswith(".mjpg"):
            camera_id = Path(parsed.path).stem
            return self._stream_camera(camera_id)
        if parsed.path == "/api/calibration/state":
            return self._send_json(self._calibration_state_payload())
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
    parser.add_argument("--output-root", default="outputs/live_runs/file_sanity")
    parser.add_argument("--scene-calibration-config", default=DEFAULT_SCENE_CALIBRATION_PATH)
    parser.add_argument("--stream-target-fps", type=float, default=15.0)
    parser.add_argument("--presentation-mode", choices=["simultaneous", "sequential"], default="simultaneous")
    parser.add_argument("--camera-sequence", default="C1,C2,C3,C4")
    parser.add_argument("--camera-segment-sec", type=float, default=10.0)
    parser.add_argument("--travel-gap-sec", type=float, default=3.0)
    parser.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", "New Dataset"))
    parser.add_argument("--demo-pair-id", default="", help="Clip pair to show in camera streams, e.g. a1, a3, b1. Use auto to infer from output-root basename.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    output_root = resolve_path(project_root, args.output_root)
    web_root = Path(__file__).resolve().parent / "web_demo"
    scene_calibration_path = resolve_path(project_root, args.scene_calibration_config)
    calibration, _runtime = load_scene_calibration(
        config_path=str(scene_calibration_path),
        base_dir=project_root,
        required=False,
    )
    demo_pair_id = resolve_demo_pair_id(output_root, args.demo_pair_id)
    applied_video_sources = apply_demo_pair_video_sources(
        calibration,
        project_root,
        args.dataset_root,
        demo_pair_id,
    )
    runtime_cameras = {}
    frame_hub = FrameBufferHub()
    demo_clock = None
    if args.presentation_mode == "sequential":
        demo_clock = SequentialDemoClock(
            [item.strip() for item in args.camera_sequence.split(",") if item.strip()],
            segment_sec=args.camera_segment_sec,
            travel_gap_sec=args.travel_gap_sec,
        )
    workers = []
    for camera_id, camera_cfg in sorted(((calibration.get("cameras", {}) or {}).items())):
        try:
            frame_size_ref = camera_cfg.get("frame_size_ref", {}) or {}
            width = int(frame_size_ref.get("width", 0) or 0)
            height = int(frame_size_ref.get("height", 0) or 0)
            if width <= 0 or height <= 0:
                source_type, source_value = resolve_preview_source(project_root, camera_cfg)
                probe = probe_frame_from_source(source_type, source_value, frame_idx=0)
                height, width = probe.shape[:2]
            runtime_cameras[camera_id] = build_runtime_camera_calibration(camera_cfg, width, height)
            worker = OfflineReplayFrameWorker(
                camera_id,
                camera_cfg,
                runtime_cameras[camera_id],
                project_root,
                output_root,
                frame_hub,
                target_fps=args.stream_target_fps,
                demo_clock=demo_clock,
            )
            workers.append(worker)
            worker.start()
        except Exception as exc:
            frame_hub.set_error(camera_id, exc)

    def handler(*handler_args, **handler_kwargs):
        return LiveDemoRequestHandler(
            *handler_args,
            web_root=web_root,
            project_root=project_root,
            output_root=output_root,
            scene_calibration_path=scene_calibration_path,
            frame_hub=frame_hub,
            **handler_kwargs,
        )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"LIVE_DEMO_UI=http://{args.host}:{args.port}")
    print(f"LIVE_DEMO_OUTPUT_ROOT={output_root}")
    print(f"SCENE_CALIBRATION_CONFIG={scene_calibration_path}")
    print(f"STREAM_TARGET_FPS={args.stream_target_fps}")
    print(f"PRESENTATION_MODE={args.presentation_mode}")
    print(f"DEMO_PAIR_ID={demo_pair_id or 'from_calibration_preview_source'}")
    for camera_id, source_path in sorted(applied_video_sources.items()):
        print(f"DEMO_VIDEO_SOURCE[{camera_id}]={source_path}")
    if demo_clock:
        print(f"CAMERA_SEQUENCE={','.join(demo_clock.camera_sequence)} SEGMENT_SEC={demo_clock.segment_sec} TRAVEL_GAP_SEC={demo_clock.travel_gap_sec}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=2.0)
        summary_path = output_root / "summaries" / "server_runtime_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "target_fps": args.stream_target_fps,
                    "camera_state": frame_hub.snapshot(),
                    "architecture": "background_frame_workers_latest_jpeg_buffer_mjpeg_stream",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        server.server_close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
