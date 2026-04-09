import json
from pathlib import Path

from run_live_event_demo_server import artifact_url, build_browser_event, is_within_root, load_latest_events


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
