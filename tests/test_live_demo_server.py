import json
from pathlib import Path

import cv2
import numpy as np

from run_live_event_demo_server import (
    _resolve_clip_video_sources,
    _resolve_single_camera_video_source,
    artifact_url,
    build_browser_event,
    is_within_root,
    load_identity_timeline,
    load_latest_events,
    render_calibration_preview,
)


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_artifact_url_encodes_path():
    url = artifact_url(r"D:\demo path\crop 01.png")
    assert url.startswith("/artifact?path=")
    assert "crop%2001.png" in url


def test_load_latest_events_adds_snapshot_urls(tmp_path):
    output_root = tmp_path / "outputs"
    events_dir = output_root / "events"
    events_dir.mkdir(parents=True)
    payload = [
        {
            "camera_id": "C5",
            "identity_type": "unknown",
            "identity_id": "UNK_0001",
            "snapshot_path": r"D:\demo\body.png",
            "head_snapshot_path": r"D:\demo\head.png",
        }
    ]
    (events_dir / "latest_events.json").write_text(json.dumps(payload), encoding="utf-8")

    events = load_latest_events(output_root)
    assert events[0]["snapshot_url"].startswith("/artifact?path=")
    assert events[0]["head_snapshot_url"].startswith("/artifact?path=")


def test_build_browser_event_keeps_pending_identity_state():
    payload = build_browser_event(
        {
            "camera_id": "C5",
            "identity_type": "pending",
            "identity_label": "Analyzing...",
            "identity_id": "",
            "snapshot_path": "",
            "head_snapshot_path": "",
            "ui_box_style": "pending_gray_dashed",
        }
    )
    assert payload["identity_type"] == "pending"
    assert payload["identity_label"] == "Analyzing..."
    assert payload["ui_box_style"] == "pending_gray_dashed"


def test_load_identity_timeline_adds_snapshot_urls(tmp_path):
    output_root = tmp_path / "outputs"
    timelines_dir = output_root / "timelines"
    timelines_dir.mkdir(parents=True)
    payload = [
        {
            "identity_id": "UNK_0001",
            "identity_label": "UNK_0001",
            "identity_status": "unknown",
            "appearance_count": 2,
            "camera_sequence": ["C1", "C2"],
            "first_seen_camera": "C1",
            "first_seen_relative_sec": 3.1,
            "last_seen_camera": "C2",
            "last_seen_relative_sec": 9.1,
            "representative_snapshot_path": r"D:\demo\body.png",
            "representative_head_snapshot_path": r"D:\demo\head.png",
            "appearances": [
                {
                    "camera_id": "C1",
                    "relative_sec": 3.1,
                    "best_body_crop": r"D:\demo\a.png",
                    "best_head_crop": r"D:\demo\a_head.png",
                },
                {
                    "camera_id": "C2",
                    "relative_sec": 9.1,
                    "best_body_crop": r"D:\demo\b.png",
                    "best_head_crop": r"D:\demo\b_head.png",
                },
            ],
        }
    ]
    (timelines_dir / "unknown_identity_timeline.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = load_identity_timeline(output_root)
    assert rows[0]["representative_snapshot_url"].startswith("/artifact?path=")
    assert rows[0]["appearances"][0]["snapshot_url"].startswith("/artifact?path=")


def test_is_within_root_rejects_parent_escape(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    inside = project_root / "outputs" / "a.png"
    inside.parent.mkdir(parents=True)
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    assert is_within_root(inside, project_root) is True
    assert is_within_root(outside, project_root) is False


def test_render_calibration_preview_can_return_clean_or_overlay_frame(tmp_path):
    preview_path = tmp_path / "preview.png"
    image = np.full((80, 120, 3), 30, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(preview_path))

    calibration = {
        "cameras": {
            "C5": {
                "preview_source_type": "file",
                "preview_source": str(preview_path),
                "processing_roi": {"polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]},
                "entry_line": {"points": [[0.2, 0.5], [0.8, 0.5]], "in_side_point": [0.5, 0.8]},
                "zones": [],
                "subzones": [],
            }
        }
    }

    clean = render_calibration_preview(calibration, "C5", tmp_path, overlay_enabled=False)
    overlay = render_calibration_preview(calibration, "C5", tmp_path, overlay_enabled=True)

    assert clean.shape == overlay.shape
    assert np.array_equal(clean, image)
    assert not np.array_equal(clean, overlay)


def test_d1_camera_1_resolves_exact_before_alias(tmp_path):
    camera_1 = tmp_path / "Camera 1"
    _touch(camera_1 / "d1.mp4")
    _touch(camera_1 / "d.mp4")
    resolved = _resolve_single_camera_video_source(camera_1, "d1", camera_folder_name="Camera 1")
    assert resolved["path"] == (camera_1 / "d1.mp4").resolve()
    assert resolved["source_note"] == "EXACT"


def test_d1_camera_1_uses_d_alias_when_exact_missing(tmp_path):
    camera_1 = tmp_path / "Camera 1"
    _touch(camera_1 / "d.mp4")
    resolved = _resolve_single_camera_video_source(camera_1, "d1", camera_folder_name="Camera 1")
    assert resolved["path"] == (camera_1 / "d.mp4").resolve()
    assert resolved["source_note"] == "ALIAS_D_TO_D1"


def test_d1_camera_2_does_not_use_d_alias(tmp_path):
    camera_2 = tmp_path / "Camera 2"
    _touch(camera_2 / "d.mp4")
    resolved = _resolve_single_camera_video_source(camera_2, "d1", camera_folder_name="Camera 2")
    assert resolved["path"] is None
    assert resolved["source_note"] == "MISSING"


def test_no_fallback_to_unrelated_pair(tmp_path):
    camera_1 = tmp_path / "Camera 1"
    _touch(camera_1 / "d2.mp4")
    resolved = _resolve_single_camera_video_source(camera_1, "d1", camera_folder_name="Camera 1")
    assert resolved["path"] is None
    assert resolved["source_note"] == "MISSING"


def test_d1_sources_map_logical_replays_and_notes(tmp_path):
    _touch(tmp_path / "Camera 1" / "d.mp4")
    _touch(tmp_path / "Camera 2" / "d1.mp4")
    sources = _resolve_clip_video_sources("d1", tmp_path)
    assert sources["C1"]["path"] == (tmp_path / "Camera 1" / "d.mp4").resolve()
    assert sources["C1"]["source_note"] == "ALIAS_D_TO_D1"
    assert sources["C2"]["path"] == (tmp_path / "Camera 2" / "d1.mp4").resolve()
    assert sources["C2"]["source_note"] == "EXACT"
    assert sources["C3"] == sources["C1"]
    assert sources["C4"] == sources["C2"]


def test_d2_sources_resolve_exact_only(tmp_path):
    _touch(tmp_path / "Camera 1" / "d2.mp4")
    _touch(tmp_path / "Camera 2" / "d2.mp4")
    sources = _resolve_clip_video_sources("d2", tmp_path)
    assert sources["C1"]["source_note"] == "EXACT"
    assert sources["C2"]["source_note"] == "EXACT"
    assert sources["C3"]["source_note"] == "EXACT"
    assert sources["C4"]["source_note"] == "EXACT"
