import argparse
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
        self.camera_segment_sec = max(1.0, float(camera_segment_sec or 12.0))
        self.gap_seconds = max(0.0, float(gap_seconds or 0.0))
        self.stream_target_fps = max(1.0, float(stream_target_fps or 15.0))
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None
        self.demo_pair_id = demo_pair_id or ""
        self.demo_start_monotonic = time.monotonic()
        self.video_sources = _resolve_clip_video_sources(self.demo_pair_id, self.dataset_root)
        self.video_paths = {
            camera_id: str(item.get("path")) if item.get("path") else None
            for camera_id, item in self.video_sources.items()
        }
        self.video_source_notes = {
            camera_id: str(item.get("source_note", "MISSING") or "MISSING")
            for camera_id, item in self.video_sources.items()
        }
        self._streamers = {
            camera_id: VideoStreamer(camera_id, video_path, target_fps=self.stream_target_fps)
            for camera_id, video_path in self.video_paths.items()
            if video_path
        }
        self._active_stream_camera = None

    def _elapsed(self):
        with self._lock:
            return max(0.0, time.monotonic() - self.demo_start_monotonic)

    def current_state(self):
        elapsed = self._elapsed()
        timeline_cursor = 0.0
        for index, camera_id in enumerate(self.sequence):
            segment_start = timeline_cursor
            segment_end = segment_start + self.camera_segment_sec
            if elapsed < segment_end:
                return {
                    "phase": "cam_playing",
                    "active_camera": camera_id,
                    "sequence": list(self.sequence),
                    "phase_start_time": segment_start,
                    "phase_elapsed_sec": elapsed - segment_start,
                    "demo_time_sec": elapsed,
                    "gap_seconds": self.gap_seconds,
                    "camera_segment_sec": self.camera_segment_sec,
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
                        "camera_segment_sec": self.camera_segment_sec,
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
            "camera_segment_sec": self.camera_segment_sec,
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
        **kwargs,
    ):
        self.web_root = web_root
        self.project_root = project_root
        self.output_root = output_root
        self.scene_calibration_path = scene_calibration_path
        self.demo_state = demo_state
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
    parser.add_argument("--camera-segment-sec", type=float, default=12.0)
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
    demo_state = DemoPlaybackState(
        presentation_mode=args.presentation_mode,
        camera_sequence=camera_sequence or list(DEFAULT_CAMERA_SEQUENCE),
        camera_segment_sec=args.camera_segment_sec,
        gap_seconds=args.travel_gap_sec,
        stream_target_fps=args.stream_target_fps,
        dataset_root=dataset_root,
        demo_pair_id=args.demo_pair_id,
    )

    def handler(*handler_args, **handler_kwargs):
        return LiveDemoRequestHandler(
            *handler_args,
            web_root=web_root,
            project_root=project_root,
            output_root=output_root,
            scene_calibration_path=scene_calibration_path,
            demo_state=demo_state,
            **handler_kwargs,
        )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"LIVE_DEMO_UI=http://{args.host}:{args.port}")
    print(f"LIVE_DEMO_OUTPUT_ROOT={output_root}")
    print(f"SCENE_CALIBRATION_CONFIG={scene_calibration_path}")
    print(f"DEMO_PAIR_ID={args.demo_pair_id}")
    print(f"PRESENTATION_MODE={args.presentation_mode}")
    print(f"CAMERA_SEQUENCE={','.join(camera_sequence or DEFAULT_CAMERA_SEQUENCE)}")
    print(f"CAMERA_SEGMENT_SEC={args.camera_segment_sec}")
    print(f"TRAVEL_GAP_SEC={args.travel_gap_sec}")
    print(f"STREAM_TARGET_FPS={args.stream_target_fps}")
    if dataset_root:
        print(f"DATASET_ROOT={dataset_root}")
    _print_video_path_resolution(demo_state.video_paths, dataset_root, source_notes=demo_state.video_source_notes)
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
