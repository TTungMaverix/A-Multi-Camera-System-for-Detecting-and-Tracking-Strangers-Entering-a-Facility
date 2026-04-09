from pathlib import Path

import numpy as np

from live_pipeline.orchestrator import LatestFrameReader, SimpleTrack, _build_event_payload, load_live_config


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def isOpened(self):
        return True

    def release(self):
        self.released = True


def test_live_pipeline_file_config_has_four_sources():
    config_path = Path("insightface_demo_assets/runtime/config/live_pipeline_demo.file_sanity.yaml")
    config = load_live_config(config_path)
    assert sorted(config["sources"].keys()) == ["C3", "C5", "C6", "C7"]
    assert config["live"]["latest_frame_only"] is True


def test_latest_frame_reader_counts_dropped_frames():
    frames = [np.full((32, 32, 3), fill_value=index, dtype=np.uint8) for index in range(6)]
    reader = LatestFrameReader(FakeCapture(frames), latest_frame_only=True, reconnect_sleep_sec=0.01, max_reconnect_attempts=1).start()
    try:
        import time

        time.sleep(0.05)
        frame, frame_idx, dropped = reader.get(timeout_sec=0.2)
    finally:
        reader.stop()

    assert frame is not None
    assert frame_idx is not None
    assert dropped >= 1


def test_live_event_payload_contains_zone_and_subzone(tmp_path):
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    track = SimpleTrack(
        track_id=7,
        bbox={"xmin": 20, "ymin": 15, "xmax": 90, "ymax": 110},
        hits=3,
        prev_foot={"x": 50.0, "y": 80.0},
        last_foot={"x": 55.0, "y": 108.0},
        last_frame_idx=12,
    )
    transition_map = {
        "cameras": {
            "C5": {
                "default_zone_id": "c5_entry_main",
                "default_subzone_id": "c5_inner_exit",
                "zones": [
                    {
                        "zone_id": "c5_entry_main",
                        "zone_type": "entry",
                        "polygon": [[0, 0], [160, 0], [160, 120], [0, 120]],
                    }
                ],
                "subzones": [
                    {
                        "subzone_id": "c5_inner_exit",
                        "parent_zone_id": "c5_entry_main",
                        "subzone_type": "exit",
                        "priority": 10,
                        "polygon": [[0, 0], [160, 0], [160, 120], [0, 120]],
                    }
                ],
            }
        }
    }
    payload = _build_event_payload(
        "C5",
        "entry",
        track,
        frame,
        frame_idx=12,
        relative_sec=1.25,
        head_cfg={"top_ratio": 0.02, "bottom_ratio": 0.45, "side_ratio": 0.18},
        transition_map=transition_map,
        output_dir=tmp_path,
    )

    assert payload is not None
    assert payload["event_type"] == "ENTRY_IN"
    assert payload["zone_id"] == "c5_entry_main"
    assert payload["subzone_id"] == "c5_inner_exit"
    assert payload["direction"] == "IN"
    assert Path(payload["best_body_crop"]).exists()
