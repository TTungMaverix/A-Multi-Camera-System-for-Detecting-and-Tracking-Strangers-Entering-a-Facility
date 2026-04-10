from pathlib import Path

import cv2
import numpy as np
import yaml

from offline_pipeline.event_builder import FrameSourceCache, build_entry_anchor_packets, build_entry_events, select_best_record
from offline_pipeline.orchestrator import load_pipeline_config


def _write_dummy_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 120, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def test_offline_pipeline_example_has_four_video_sources():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.example.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "wildtrack_gt_annotations"
    assert sorted(config["dataset"]["video_sources"].keys()) == ["C3", "C5", "C6", "C7"]


def test_entry_event_builder_emits_in_event_with_zone_and_subzone(tmp_path):
    image_path = tmp_path / "frame.png"
    _write_dummy_png(image_path)
    track_rows_by_camera = {
        "C5": [
            {
                "camera_id": "C5",
                "role": "entry",
                "local_track_id": "1",
                "global_gt_id": 1,
                "frame_id": 0,
                "relative_sec": 0.0,
                "xmin": 20,
                "ymin": 20,
                "xmax": 80,
                "ymax": 70,
                "width": 60,
                "height": 50,
                "area": 3000,
                "foot_x": 50.0,
                "foot_y": 40.0,
                "image_rel_path": "unused.png",
                "source_image_path": str(image_path),
            },
            {
                "camera_id": "C5",
                "role": "entry",
                "local_track_id": "1",
                "global_gt_id": 1,
                "frame_id": 1,
                "relative_sec": 0.1,
                "xmin": 20,
                "ymin": 20,
                "xmax": 85,
                "ymax": 90,
                "width": 65,
                "height": 70,
                "area": 4550,
                "foot_x": 55.0,
                "foot_y": 90.0,
                "image_rel_path": "unused.png",
                "source_image_path": str(image_path),
            },
        ]
    }
    wildtrack_config = {
        "selected_cameras": ["C5"],
        "best_shot_window_frames": 5,
        "head_crop": {"top_ratio": 0.02, "bottom_ratio": 0.45, "side_ratio": 0.18},
        "direction_filter": {
            "history_window": 4,
            "minimum_points": 2,
            "minimum_inward_motion_px": 5.0,
            "minimum_inside_ratio": 1.0,
            "require_zone_transition": False,
            "allow_line_cross_only_when_history_short": True,
            "inward_zone_types": ["exit", "interior", "overlap", "transit"],
            "outward_zone_types": ["entry", "outer", "approach"],
        },
        "cameras": {
            "C5": {
                "role": "entry",
                "entry_line": [[0, 60], [100, 60]],
                "in_side_point": [50, 90],
                "direction_filter": {
                    "history_window": 4,
                    "minimum_points": 2,
                    "minimum_inward_motion_px": 5.0,
                    "minimum_inside_ratio": 1.0,
                    "require_zone_transition": False,
                    "allow_line_cross_only_when_history_short": True,
                    "inward_zone_types": ["exit", "interior", "overlap", "transit"],
                    "outward_zone_types": ["entry", "outer", "approach"],
                },
            }
        },
    }
    transition_map = {
        "cameras": {
            "C5": {
                "default_zone_id": "c5_entry_main",
                "default_subzone_id": "c5_inner_exit",
                "zones": [{"zone_id": "c5_entry_main", "zone_type": "entry", "polygon": [[0, 0], [100, 0], [100, 120], [0, 120]]}],
                "subzones": [{"subzone_id": "c5_inner_exit", "parent_zone_id": "c5_entry_main", "subzone_type": "exit", "polygon": [[0, 0], [100, 0], [100, 120], [0, 120]], "priority": 10}],
            }
        }
    }
    cache = FrameSourceCache()
    try:
        events, queue_rows, audit_rows = build_entry_events(
            tmp_path,
            wildtrack_config,
            track_rows_by_camera,
            cache,
            tmp_path / "crops",
            transition_map,
        )
    finally:
        cache.close()

    assert len(events) == 1
    assert len(queue_rows) == 1
    assert events[0]["direction"] == "IN"
    assert events[0]["zone_id"] == "c5_entry_main"
    assert events[0]["subzone_id"] == "c5_inner_exit"
    assert queue_rows[0]["best_head_crop"].endswith(".png")
    assert audit_rows[0]["subzone_id"] == "c5_inner_exit"


def test_entry_anchor_packet_schema_is_lightweight():
    rows = [
        {
            "camera_id": "C5",
            "role": "entry",
            "local_track_id": "1",
            "global_gt_id": 1,
            "frame_id": 0,
            "relative_sec": 0.0,
            "xmin": 20,
            "ymin": 20,
            "xmax": 80,
            "ymax": 70,
            "width": 60,
            "height": 50,
            "area": 3000,
            "foot_x": 50.0,
            "foot_y": 40.0,
            "video_path": "cam5.mp4",
            "image_rel_path": "Image_subsets/C5/00000000.png",
        },
        {
            "camera_id": "C5",
            "role": "entry",
            "local_track_id": "1",
            "global_gt_id": 1,
            "frame_id": 1,
            "relative_sec": 0.1,
            "xmin": 20,
            "ymin": 20,
            "xmax": 85,
            "ymax": 90,
            "width": 65,
            "height": 70,
            "area": 4550,
            "foot_x": 55.0,
            "foot_y": 90.0,
            "video_path": "cam5.mp4",
            "image_rel_path": "Image_subsets/C5/00000001.png",
        },
    ]
    packets = build_entry_anchor_packets(
        "C5",
        {
            "role": "entry",
            "entry_line": [[0, 60], [100, 60]],
            "in_side_point": [50, 90],
            "direction_filter": {
                "history_window": 4,
                "minimum_points": 2,
                "minimum_inward_motion_px": 5.0,
                "minimum_inside_ratio": 1.0,
                "require_zone_transition": False,
                "allow_line_cross_only_when_history_short": True,
                "inward_zone_types": ["exit", "interior", "overlap", "transit"],
                "outward_zone_types": ["entry", "outer", "approach"],
            },
        },
        rows,
        20,
        {
            "cameras": {
                "C5": {
                    "default_zone_id": "c5_entry_main",
                    "default_subzone_id": "c5_inner_exit",
                    "zones": [{"zone_id": "c5_entry_main", "zone_type": "entry", "polygon": [[0, 0], [100, 0], [100, 120], [0, 120]]}],
                    "subzones": [{"subzone_id": "c5_inner_exit", "parent_zone_id": "c5_entry_main", "subzone_type": "exit", "polygon": [[0, 0], [100, 0], [100, 120], [0, 120]], "priority": 10}],
                }
            }
        },
    )
    assert len(packets) == 1
    assert packets[0]["packet_type"] == "entry_anchor"
    assert packets[0]["direction"] == "IN"
    assert packets[0]["crop_reference"]["video_path"] == "cam5.mp4"


def test_line_aware_best_shot_prefers_exit_subzone_after_anchor():
    records = [
        {
            "camera_id": "C5",
            "frame_id": 10,
            "area": 50000,
            "foot_x": 10.0,
            "foot_y": 10.0,
        },
        {
            "camera_id": "C5",
            "frame_id": 14,
            "area": 30000,
            "foot_x": 80.0,
            "foot_y": 80.0,
        },
    ]
    transition_map = {
        "cameras": {
            "C5": {
                "default_zone_id": "c5_entry_main",
                "default_subzone_id": "c5_outer_entry",
                "zones": [{"zone_id": "c5_entry_main", "zone_type": "entry", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
                "subzones": [
                    {"subzone_id": "c5_outer_entry", "parent_zone_id": "c5_entry_main", "subzone_type": "entry", "polygon": [[0, 0], [40, 0], [40, 40], [0, 40]], "priority": 5},
                    {"subzone_id": "c5_inner_exit", "parent_zone_id": "c5_entry_main", "subzone_type": "exit", "polygon": [[60, 60], [100, 60], [100, 100], [60, 100]], "priority": 10},
                ],
            }
        }
    }
    selected, meta = select_best_record(
        records,
        frame_min=10,
        frame_max=20,
        camera_id="C5",
        transition_map=transition_map,
        best_shot_cfg={
            "enabled": True,
            "preferred_subzone_types": ["exit", "interior", "overlap", "transit", "entry"],
            "minimum_frames_after_anchor": {"entry": 3},
        },
        anchor_frame_id=10,
        relation_type="entry",
    )
    assert selected["frame_id"] == 14
    assert meta["best_shot_subzone_id"] == "c5_inner_exit"
    assert meta["best_shot_strategy"] == "line_aware_subzone_priority"
