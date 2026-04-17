import argparse
import json
import mimetypes
import sys
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


DEFAULT_SCENE_CALIBRATION_PATH = "insightface_demo_assets/runtime/config/manual_scene_calibration.wildtrack_4cam_phase.yaml"


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


class LiveDemoRequestHandler(SimpleHTTPRequestHandler):
    server_version = "LiveDemoHTTP/0.2"

    def __init__(self, *args, web_root=None, project_root=None, output_root=None, scene_calibration_path=None, **kwargs):
        self.web_root = web_root
        self.project_root = project_root
        self.output_root = output_root
        self.scene_calibration_path = scene_calibration_path
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

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/latest-events":
            return self._send_json({"events": load_latest_events(self.output_root)})
        if parsed.path == "/api/summary":
            return self._send_json(load_live_summary(self.output_root))
        if parsed.path == "/api/timeline":
            return self._send_json({"identities": load_identity_timeline(self.output_root)})
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
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    output_root = resolve_path(project_root, args.output_root)
    web_root = Path(__file__).resolve().parent / "web_demo"
    scene_calibration_path = resolve_path(project_root, args.scene_calibration_config)

    def handler(*handler_args, **handler_kwargs):
        return LiveDemoRequestHandler(
            *handler_args,
            web_root=web_root,
            project_root=project_root,
            output_root=output_root,
            scene_calibration_path=scene_calibration_path,
            **handler_kwargs,
        )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"LIVE_DEMO_UI=http://{args.host}:{args.port}")
    print(f"LIVE_DEMO_OUTPUT_ROOT={output_root}")
    print(f"SCENE_CALIBRATION_CONFIG={scene_calibration_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
