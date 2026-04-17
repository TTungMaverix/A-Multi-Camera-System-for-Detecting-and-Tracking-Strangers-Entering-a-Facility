import json
from pathlib import Path

from run_live_event_demo_server import artifact_url, build_browser_event, is_within_root, load_identity_timeline, load_latest_events


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
                {"camera_id": "C1", "relative_sec": 3.1, "best_body_crop": r"D:\demo\a.png", "best_head_crop": r"D:\demo\a_head.png"},
                {"camera_id": "C2", "relative_sec": 9.1, "best_body_crop": r"D:\demo\b.png", "best_head_crop": r"D:\demo\b_head.png"},
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
