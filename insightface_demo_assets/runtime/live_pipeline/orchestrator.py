import argparse
import json
import multiprocessing as mp
import queue as queue_module
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from association_core import (
    assign_model_identities,
    build_topology_index,
    load_association_policy,
    load_camera_transition_map,
    resolve_spatial_context,
)
from offline_pipeline.event_builder import (
    get_head_rect,
    test_entry_crossing,
    test_point_in_polygon,
    write_image_unicode,
)
from run_face_resolution_demo import (
    CONFIG_DEFAULT,
    analyze_event_crops,
    build_gallery_embeddings,
    load_json,
    read_csv,
    save_json,
)


def resolve_path(project_root: Path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_live_config(config_path: Path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "live_pipeline" in payload:
        return payload["live_pipeline"]
    return payload


def build_face_runtime_config(live_config, project_root: Path):
    face_demo_config_path = resolve_path(project_root, live_config["face_demo_config"])
    face_demo_config = load_json(face_demo_config_path)
    runtime_config = dict(face_demo_config)
    if live_config.get("association_policy_config"):
        runtime_config["association_policy_config"] = str(
            resolve_path(project_root, live_config["association_policy_config"])
        )
    if live_config.get("camera_transition_map_config"):
        runtime_config["camera_transition_map_config"] = str(
            resolve_path(project_root, live_config["camera_transition_map_config"])
        )
    if live_config.get("known_gallery", {}).get("manifest_csv"):
        runtime_config["known_face_manifest_csv"] = str(
            resolve_path(project_root, live_config["known_gallery"]["manifest_csv"])
        )
    if live_config.get("known_gallery", {}).get("gallery_root"):
        runtime_config["known_face_gallery_root"] = str(
            resolve_path(project_root, live_config["known_gallery"]["gallery_root"])
        )
    return runtime_config


def _open_capture(source_cfg):
    source_type = source_cfg.get("source_type", "file")
    source_value = source_cfg.get("source", "")
    if source_type == "webcam":
        return cv2.VideoCapture(int(source_value))
    return cv2.VideoCapture(source_value)


def _append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_queue_put(queue, packet, timeout_sec=0.5):
    try:
        queue.put(packet, timeout=timeout_sec)
        return True
    except queue_module.Full:
        return False


class LatestFrameReader:
    def __init__(self, capture, latest_frame_only=True, reconnect_sleep_sec=1.0, max_reconnect_attempts=3, source_cfg=None):
        self.capture = capture
        self.latest_frame_only = latest_frame_only
        self.reconnect_sleep_sec = reconnect_sleep_sec
        self.max_reconnect_attempts = max_reconnect_attempts
        self.source_cfg = source_cfg or {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._frame = None
        self._frame_seq = -1
        self._last_delivered = -1
        self._dropped_frames = 0
        self._eof = False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _reconnect(self):
        for _ in range(self.max_reconnect_attempts):
            time.sleep(self.reconnect_sleep_sec)
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = _open_capture(self.source_cfg)
            if self.capture.isOpened():
                return True
        return False

    def _run(self):
        reconnectable = self.source_cfg.get("source_type") in {"rtsp", "webcam"}
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                if reconnectable and self._reconnect():
                    continue
                self._eof = True
                break
            with self._lock:
                if self.latest_frame_only and self._frame is not None and self._frame_seq != self._last_delivered:
                    self._dropped_frames += 1
                self._frame = frame
                self._frame_seq += 1

    def get(self, timeout_sec=1.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._frame_seq != self._last_delivered:
                    self._last_delivered = self._frame_seq
                    return self._frame.copy(), self._frame_seq, self._dropped_frames
            if self._eof:
                return None, None, self._dropped_frames
            time.sleep(0.01)
        return None, None, self._dropped_frames

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.capture.release()
        except Exception:
            pass


class LightweightPersonDetector:
    def __init__(self, win_stride=(8, 8), padding=(8, 8), scale=1.05, hit_threshold=0.0):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold

    def detect(self, frame, resize_width=960):
        height, width = frame.shape[:2]
        if width > resize_width:
            ratio = resize_width / float(width)
            resized = cv2.resize(frame, (resize_width, int(round(height * ratio))), interpolation=cv2.INTER_LINEAR)
        else:
            ratio = 1.0
            resized = frame
        rects, _weights = self.hog.detectMultiScale(
            resized,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            hitThreshold=self.hit_threshold,
        )
        detections = []
        for (x, y, w, h) in rects:
            detections.append(
                {
                    "xmin": int(round(x / ratio)),
                    "ymin": int(round(y / ratio)),
                    "xmax": int(round((x + w) / ratio)),
                    "ymax": int(round((y + h) / ratio)),
                }
            )
        return detections


@dataclass
class SimpleTrack:
    track_id: int
    bbox: dict
    hits: int = 1
    missed: int = 0
    emitted: bool = False
    prev_foot: dict | None = None
    last_foot: dict | None = None
    last_frame_idx: int = 0


class CentroidTracker:
    def __init__(self, max_distance_px=140.0, max_missed_frames=10):
        self.max_distance_px = float(max_distance_px)
        self.max_missed_frames = int(max_missed_frames)
        self.next_track_id = 1
        self.tracks = {}

    @staticmethod
    def _center(bbox):
        return ((bbox["xmin"] + bbox["xmax"]) / 2.0, (bbox["ymin"] + bbox["ymax"]) / 2.0)

    def update(self, detections, frame_idx):
        updated_ids = set()
        assigned_detections = set()
        track_items = list(self.tracks.items())
        for track_id, track in track_items:
            best_index = None
            best_dist = None
            cx1, cy1 = self._center(track.bbox)
            for index, det in enumerate(detections):
                if index in assigned_detections:
                    continue
                cx2, cy2 = self._center(det)
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                if dist <= self.max_distance_px and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_index = index
            if best_index is None:
                track.missed += 1
                if track.missed > self.max_missed_frames:
                    del self.tracks[track_id]
                continue
            det = detections[best_index]
            assigned_detections.add(best_index)
            track.hits += 1
            track.missed = 0
            track.prev_foot = track.last_foot
            track.last_foot = {
                "x": (det["xmin"] + det["xmax"]) / 2.0,
                "y": float(det["ymax"]),
            }
            track.bbox = det
            track.last_frame_idx = frame_idx
            updated_ids.add(track_id)
        for index, det in enumerate(detections):
            if index in assigned_detections:
                continue
            track = SimpleTrack(
                track_id=self.next_track_id,
                bbox=det,
                hits=1,
                missed=0,
                emitted=False,
                prev_foot=None,
                last_foot={"x": (det["xmin"] + det["xmax"]) / 2.0, "y": float(det["ymax"])},
                last_frame_idx=frame_idx,
            )
            self.tracks[self.next_track_id] = track
            updated_ids.add(self.next_track_id)
            self.next_track_id += 1
        return [self.tracks[track_id] for track_id in sorted(updated_ids) if track_id in self.tracks]


def _crop_rect(frame, rect):
    height, width = frame.shape[:2]
    xmin = max(0, int(rect["xmin"]))
    ymin = max(0, int(rect["ymin"]))
    xmax = min(width, int(rect["xmax"]))
    ymax = min(height, int(rect["ymax"]))
    if xmax <= xmin or ymax <= ymin:
        return None
    return frame[ymin:ymax, xmin:xmax]


def _build_event_payload(camera_id, role, track, frame, frame_idx, relative_sec, head_cfg, transition_map, output_dir):
    bbox = track.bbox
    body_crop = _crop_rect(frame, bbox)
    if body_crop is None or body_crop.size == 0:
        return None
    head_rect = get_head_rect(
        {
            "xmin": bbox["xmin"],
            "ymin": bbox["ymin"],
            "width": bbox["xmax"] - bbox["xmin"],
            "height": bbox["ymax"] - bbox["ymin"],
        },
        head_cfg,
        frame.shape[1],
        frame.shape[0],
    )
    head_crop = frame[head_rect["y"] : head_rect["y"] + head_rect["height"], head_rect["x"] : head_rect["x"] + head_rect["width"]]
    spatial = resolve_spatial_context(camera_id, track.last_foot["x"], track.last_foot["y"], transition_map)
    event_type = "ENTRY_IN" if role == "entry" else "FOLLOWUP_OBSERVATION"
    crop_dir = output_dir / "snapshots" / camera_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    event_id = f"LIVE_{camera_id}_{track.track_id:04d}_{frame_idx:06d}"
    body_path = crop_dir / f"{event_id}_body.png"
    head_path = crop_dir / f"{event_id}_head.png"
    write_image_unicode(body_path, body_crop)
    if head_crop.size:
        write_image_unicode(head_path, head_crop)
    else:
        write_image_unicode(head_path, body_crop)
    return {
        "event_id": event_id,
        "event_type": event_type,
        "camera_id": camera_id,
        "frame_id": frame_idx,
        "relative_sec": round(relative_sec, 3),
        "global_gt_id": "",
        "local_track_id": f"{camera_id}_{track.track_id}",
        "anchor_camera_id": "",
        "anchor_relative_sec": "",
        "relation_type": "entry" if role == "entry" else "overlap",
        "zone_id": spatial["zone_id"],
        "zone_type": spatial.get("zone_type", ""),
        "zone_reason": spatial.get("zone_reason", ""),
        "zone_fallback_used": spatial.get("zone_fallback_used", False),
        "subzone_id": spatial["subzone_id"],
        "subzone_type": spatial.get("subzone_type", ""),
        "subzone_reason": spatial.get("subzone_reason", ""),
        "subzone_fallback_used": spatial.get("subzone_fallback_used", False),
        "matched_zone_region_id": spatial.get("matched_zone_region_id", ""),
        "matched_subzone_region_id": spatial.get("matched_subzone_region_id", ""),
        "assignment_point_x": spatial.get("assignment_point_x", ""),
        "assignment_point_y": spatial.get("assignment_point_y", ""),
        "best_head_crop": str(head_path),
        "best_body_crop": str(body_path),
        "bbox_xmin": bbox["xmin"],
        "bbox_ymin": bbox["ymin"],
        "bbox_xmax": bbox["xmax"],
        "bbox_ymax": bbox["ymax"],
        "bbox_width": bbox["xmax"] - bbox["xmin"],
        "bbox_height": bbox["ymax"] - bbox["ymin"],
        "bbox_area": max(1, (bbox["xmax"] - bbox["xmin"]) * (bbox["ymax"] - bbox["ymin"])),
        "foot_x": track.last_foot["x"],
        "foot_y": track.last_foot["y"],
        "direction": "IN" if role == "entry" else "FOLLOWUP",
    }


def _worker_should_emit(camera_cfg, track, frame_idx, transition_map, output_dir, frame, head_cfg, start_time, mode_cfg):
    role = camera_cfg.get("role", "followup")
    if role == "entry":
        line = camera_cfg.get("entry_line")
        in_side_point = camera_cfg.get("in_side_point")
        if not line or not in_side_point or track.prev_foot is None or track.last_foot is None:
            return None
        if track.emitted:
            return None
        if test_entry_crossing(track.prev_foot, track.last_foot, line, in_side_point, mode_cfg["line_distance_threshold"]):
            track.emitted = True
            relative_sec = time.time() - start_time
            return _build_event_payload(camera_cfg["camera_id"], role, track, frame, frame_idx, relative_sec, mode_cfg["head_crop"], transition_map, output_dir)
        return None

    if track.emitted or track.hits < mode_cfg["followup_emit_min_hits"]:
        return None
    track.emitted = True
    relative_sec = time.time() - start_time
    return _build_event_payload(camera_cfg["camera_id"], role, track, frame, frame_idx, relative_sec, mode_cfg["head_crop"], transition_map, output_dir)


def _filter_detections_by_roi(detections, camera_cfg):
    polygon = camera_cfg.get("entry_roi") or camera_cfg.get("track_roi") or []
    if not polygon:
        return detections
    kept = []
    for det in detections:
        foot_x = (det["xmin"] + det["xmax"]) / 2.0
        foot_y = float(det["ymax"])
        if test_point_in_polygon(foot_x, foot_y, polygon):
            kept.append(det)
    return kept


def live_camera_worker(worker_context, queue):
    camera_id = worker_context["camera_cfg"]["camera_id"]
    source_cfg = worker_context["source_cfg"]
    output_dir = Path(worker_context["output_dir"])
    start_time = time.time()
    reader = None
    capture = None
    detector = LightweightPersonDetector(
        scale=float(worker_context["detector_cfg"].get("scale", 1.05)),
        hit_threshold=float(worker_context["detector_cfg"].get("hit_threshold", 0.0)),
    )
    tracker = CentroidTracker(
        max_distance_px=float(worker_context["tracker_cfg"].get("max_distance_px", 140.0)),
        max_missed_frames=int(worker_context["tracker_cfg"].get("max_missed_frames", 10)),
    )
    queue_dropped_packets = 0
    try:
        if not _safe_queue_put(
            queue,
            {
                "packet_type": "worker_started",
                "camera_id": camera_id,
                "source_type": source_cfg.get("source_type", "file"),
                "latest_frame_only": bool(worker_context["live_cfg"].get("latest_frame_only", True)),
            },
            timeout_sec=float(worker_context["live_cfg"].get("queue_put_timeout_sec", 0.5)),
        ):
            queue_dropped_packets += 1
        capture = _open_capture(source_cfg)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open source for {camera_id}: {source_cfg}")
        reader = LatestFrameReader(
            capture,
            latest_frame_only=bool(worker_context["live_cfg"].get("latest_frame_only", True)),
            reconnect_sleep_sec=float(worker_context["live_cfg"].get("reconnect_sleep_sec", 1.0)),
            max_reconnect_attempts=int(worker_context["live_cfg"].get("max_reconnect_attempts", 3)),
            source_cfg=source_cfg,
        ).start()
        duration_sec = float(worker_context["live_cfg"].get("duration_sec", 0.0))
        target_fps = float(worker_context["live_cfg"].get("target_fps", 4.0))
        min_interval = 1.0 / max(target_fps, 0.1)
        last_emit = 0.0
        processed_frames = 0
        emitted_events = 0
        while True:
            if duration_sec > 0.0 and (time.time() - start_time) > duration_sec:
                break
            frame, frame_idx, dropped_frames = reader.get(timeout_sec=1.0)
            if frame is None:
                break
            now = time.time()
            if (now - last_emit) < min_interval:
                continue
            last_emit = now
            processed_frames += 1
            detections = detector.detect(frame, resize_width=int(worker_context["detector_cfg"].get("resize_width", 960)))
            detections = _filter_detections_by_roi(detections, worker_context["camera_cfg"])
            tracks = tracker.update(detections, frame_idx)
            for track in tracks:
                packet_event = _worker_should_emit(
                    worker_context["camera_cfg"],
                    track,
                    frame_idx,
                    worker_context["transition_map"],
                    output_dir,
                    frame,
                    worker_context["head_crop"],
                    start_time,
                    {
                        "followup_emit_min_hits": int(worker_context["live_cfg"].get("followup_emit_min_hits", 3)),
                        "line_distance_threshold": float(worker_context["line_threshold"]),
                        "head_crop": worker_context["head_crop"],
                    },
                )
                if packet_event is None:
                    continue
                emitted_events += 1
                if not _safe_queue_put(
                    queue,
                    {
                        "packet_type": "live_event",
                        "camera_id": camera_id,
                        "event": packet_event,
                        "dropped_frames": dropped_frames,
                    },
                    timeout_sec=float(worker_context["live_cfg"].get("queue_put_timeout_sec", 0.5)),
                ):
                    queue_dropped_packets += 1
        _safe_queue_put(
            queue,
            {
                "packet_type": "worker_summary",
                "camera_id": camera_id,
                "processed_frames": processed_frames,
                "emitted_events": emitted_events,
                "dropped_frames": reader._dropped_frames if reader else 0,
                "queue_dropped_packets": queue_dropped_packets,
            },
            timeout_sec=float(worker_context["live_cfg"].get("queue_put_timeout_sec", 0.5)),
        )
    except Exception as exc:
        _safe_queue_put(
            queue,
            {
                "packet_type": "worker_error",
                "camera_id": camera_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            timeout_sec=float(worker_context["live_cfg"].get("queue_put_timeout_sec", 0.5)),
        )
    finally:
        if reader is not None:
            reader.stop()
        elif capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        _safe_queue_put(
            queue,
            {"packet_type": "worker_done", "camera_id": camera_id},
            timeout_sec=float(worker_context["live_cfg"].get("queue_put_timeout_sec", 0.5)),
        )


def _load_known_gallery(runtime_config, app, project_root: Path, output_root: Path):
    manifest_csv = resolve_path(project_root, runtime_config["known_face_manifest_csv"])
    manifest_rows = read_csv(manifest_csv) if manifest_csv.exists() else []
    gallery_embeddings_csv = output_root / "events" / "known_face_embeddings_live.csv"
    identity_means, _rows = build_gallery_embeddings(app, manifest_rows, project_root, gallery_embeddings_csv)
    return identity_means


def _build_live_event_view(latest_row, event, latest_log, pipeline_start_time):
    event_time = pipeline_start_time + float(latest_row["relative_sec"])
    latency_sec = max(0.0, time.time() - event_time)
    return {
        "timestamp": time.time(),
        "camera_id": latest_row["camera_id"],
        "identity_type": latest_row["identity_status"],
        "identity_id": latest_row["resolved_global_id"] or latest_row["matched_known_id"] or "",
        "direction": event.get("direction", ""),
        "zone_id": latest_row.get("zone_id", ""),
        "subzone_id": latest_row.get("subzone_id", ""),
        "modality_primary": latest_row.get("modality_primary_used", ""),
        "modality_secondary": latest_row.get("modality_secondary_used", ""),
        "body_fallback_used": latest_row.get("body_fallback_used", False),
        "face_unusable_reason": latest_row.get("face_unusable_reason", ""),
        "snapshot_path": latest_row.get("best_body_crop", ""),
        "head_snapshot_path": latest_row.get("best_head_crop", ""),
        "decision_reason": latest_row.get("decision_reason", ""),
        "reason_code": latest_log.get("reason_code", ""),
        "relative_sec": latest_row["relative_sec"],
        "latency_sec": round(latency_sec, 3),
    }


def run_live_pipeline(config_path: Path):
    live_config = load_live_config(config_path)
    project_root = resolve_path(config_path.parent, live_config.get("project_root", str(CONFIG_DEFAULT.parents[2])))
    output_root = resolve_path(project_root, live_config["output_root"])
    for name in ("events", "logs", "summaries", "snapshots", "association_logs"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    runtime_config = build_face_runtime_config(live_config, project_root)
    wildtrack_config = load_json(resolve_path(project_root, live_config["wildtrack_demo_config"]))
    transition_config_path = live_config.get("camera_transition_map_config", "")
    if transition_config_path:
        transition_config_path = str(resolve_path(project_root, transition_config_path))
    transition_map, transition_runtime = load_camera_transition_map(
        wildtrack_config,
        config_path=transition_config_path,
        base_dir=project_root,
    )
    policy_config_path = live_config.get("association_policy_config", "")
    if policy_config_path:
        policy_config_path = str(resolve_path(project_root, policy_config_path))
    policy, policy_runtime = load_association_policy(
        config_path=policy_config_path,
        base_dir=project_root,
    )
    topology = build_topology_index(transition_map)
    app = __import__("insightface.app", fromlist=["FaceAnalysis"]).FaceAnalysis(
        name=runtime_config["insightface_runtime"].get("recommended_model_name", "buffalo_l"),
        root=runtime_config["insightface_runtime"].get("recommended_model_root", "C:/Users/Admin/.insightface"),
        providers=[runtime_config["insightface_runtime"].get("provider", "CPUExecutionProvider")],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    identity_means = _load_known_gallery(runtime_config, app, project_root, output_root)

    mp.set_start_method((live_config.get("execution", {}) or {}).get("start_method", "spawn"), force=True)
    queue = mp.Queue(maxsize=max(8, int((live_config.get("execution", {}) or {}).get("queue_max_size", 64))))
    workers = []
    pipeline_start_time = time.time()
    for camera_id, source_cfg in live_config["sources"].items():
        camera_cfg = dict(wildtrack_config["cameras"][camera_id])
        camera_cfg["camera_id"] = camera_id
        resolved_source_cfg = dict(source_cfg)
        if resolved_source_cfg.get("source_type", "file") == "file":
            resolved_source_cfg["source"] = str(resolve_path(project_root, resolved_source_cfg["source"]))
        ctx = {
            "camera_cfg": camera_cfg,
            "source_cfg": resolved_source_cfg,
            "output_dir": str(output_root),
            "transition_map": transition_map,
            "line_threshold": float(wildtrack_config["line_crossing_distance_threshold"]),
            "head_crop": wildtrack_config["head_crop"],
            "live_cfg": live_config.get("live", {}),
            "detector_cfg": live_config.get("detector", {}),
            "tracker_cfg": live_config.get("tracker", {}),
        }
        process = mp.Process(target=live_camera_worker, args=(ctx, queue), name=f"live_worker_{camera_id}")
        process.start()
        workers.append(process)

    done = set()
    analyzed_events = []
    logs = []
    latest_state = []
    worker_summaries = {}
    errors = []
    total_packets = 0
    latency_values = []
    counters = {
        "known_event_count": 0,
        "unknown_event_count": 0,
        "body_fallback_used_count": 0,
        "dropped_frames_total": 0,
        "queue_dropped_packets_total": 0,
    }
    live_jsonl = output_root / "events" / "live_event_stream.jsonl"
    live_jsonl.write_text("", encoding="utf-8")
    resolved_jsonl = output_root / "events" / "resolved_live_events.jsonl"
    resolved_jsonl.write_text("", encoding="utf-8")
    decision_jsonl = output_root / "association_logs" / "live_decision_stream.jsonl"
    decision_jsonl.write_text("", encoding="utf-8")
    while len(done) < len(workers):
        packet = queue.get()
        total_packets += 1
        logs.append(packet)
        packet_type = packet["packet_type"]
        if packet_type == "live_event":
            event = packet["event"]
            analyzed = analyze_event_crops(app, [event])[0]
            analyzed_events.append(analyzed)
            analyzed_events.sort(key=lambda item: float(item["event"]["relative_sec"]))
            resolved_rows, _profiles, _trace_rows, debug = assign_model_identities(
                analyzed_events,
                identity_means,
                topology,
                unknown_prefix=runtime_config["unknown_handling"].get("seed_prefix", "UNK"),
                unknown_start=int(runtime_config["unknown_handling"].get("start_index", 1)),
                policy=policy,
                return_debug_bundle=True,
            )
            latest_row = resolved_rows[-1]
            latest_log = debug["decision_logs"][-1]
            live_event = _build_live_event_view(latest_row, event, latest_log, pipeline_start_time)
            latest_state.append(live_event)
            latency_values.append(float(live_event["latency_sec"]))
            if live_event["identity_type"] == "known":
                counters["known_event_count"] += 1
            else:
                counters["unknown_event_count"] += 1
            if live_event["body_fallback_used"]:
                counters["body_fallback_used_count"] += 1
            _append_jsonl(live_jsonl, live_event)
            _append_jsonl(resolved_jsonl, latest_row)
            _append_jsonl(decision_jsonl, latest_log)
            save_json(output_root / "events" / "latest_events.json", latest_state[-50:])
            save_json(output_root / "association_logs" / "latest_decision_log.json", latest_log)
            print(
                f"LIVE_EVENT camera={live_event['camera_id']} type={live_event['identity_type']} "
                f"id={live_event['identity_id']} modality={live_event['modality_primary']} "
                f"body_fallback={str(live_event['body_fallback_used']).lower()} reason={live_event['decision_reason']} "
                f"latency={live_event['latency_sec']:.3f}s"
            )
        elif packet_type == "worker_summary":
            worker_summaries[packet["camera_id"]] = {
                "processed_frames": packet["processed_frames"],
                "emitted_events": packet["emitted_events"],
                "dropped_frames": packet["dropped_frames"],
                "queue_dropped_packets": packet.get("queue_dropped_packets", 0),
            }
            counters["dropped_frames_total"] += int(packet.get("dropped_frames", 0))
            counters["queue_dropped_packets_total"] += int(packet.get("queue_dropped_packets", 0))
        elif packet_type == "worker_error":
            errors.append(packet)
        elif packet_type == "worker_done":
            done.add(packet["camera_id"])

    for process in workers:
        process.join()

    summary = {
        "pipeline_name": live_config.get("pipeline_name", "live_multicam_demo"),
        "config_path": str(config_path),
        "output_root": str(output_root),
        "total_packets": total_packets,
        "live_event_count": len(latest_state),
        "known_event_count": counters["known_event_count"],
        "unknown_event_count": counters["unknown_event_count"],
        "body_fallback_used_count": counters["body_fallback_used_count"],
        "dropped_frames_total": counters["dropped_frames_total"],
        "queue_dropped_packets_total": counters["queue_dropped_packets_total"],
        "avg_latency_sec": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0.0,
        "max_latency_sec": round(max(latency_values), 3) if latency_values else 0.0,
        "worker_summaries": worker_summaries,
        "error_count": len(errors),
        "errors": errors,
        "association_policy_runtime": policy_runtime,
        "camera_transition_map_runtime": transition_runtime,
    }
    save_json(output_root / "summaries" / "live_pipeline_summary.json", summary)
    save_json(output_root / "logs" / "live_worker_packets.json", logs)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run the lightweight live multi-camera stranger demo.")
    parser.add_argument("--config", required=True, help="Path to live pipeline YAML config.")
    return parser.parse_args()


def cli():
    args = parse_args()
    run_live_pipeline(Path(args.config).resolve())


if __name__ == "__main__":
    cli()
