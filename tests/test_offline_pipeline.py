from pathlib import Path

import cv2
import numpy as np
import yaml

from offline_pipeline.event_builder import (
    FrameSourceCache,
    build_entry_anchor_packets,
    build_entry_events,
    clone_track_rows_for_virtual_camera,
    load_inference_track_cache,
    link_short_gap_tracklets,
    write_inference_track_cache,
    select_best_record,
)
from offline_pipeline.orchestrator import load_pipeline_config
from scene_calibration import filter_boxes_by_processing_roi


def _write_dummy_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 120, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def _track_row(track_id, frame_id, xmin, ymin, xmax, ymax, camera_id="C6"):
    width = xmax - xmin
    height = ymax - ymin
    return {
        "camera_id": camera_id,
        "role": "entry",
        "local_track_id": str(track_id),
        "global_gt_id": int(track_id),
        "frame_id": int(frame_id),
        "frame_key": f"{int(frame_id):08d}",
        "relative_sec": round(float(frame_id) / 10.0, 3),
        "xmin": int(xmin),
        "ymin": int(ymin),
        "xmax": int(xmax),
        "ymax": int(ymax),
        "width": int(width),
        "height": int(height),
        "area": int(width * height),
        "center_x": round((xmin + xmax) / 2.0, 2),
        "center_y": round((ymin + ymax) / 2.0, 2),
        "foot_x": round((xmin + xmax) / 2.0, 2),
        "foot_y": float(ymax),
        "image_rel_path": f"VideoReplay\\{camera_id}\\{int(frame_id):08d}.png",
        "video_path": "cam6.mp4",
        "frame_source_mode": "video_files",
        "track_provider": "ultralytics_yolo_bytetrack",
    }


def test_offline_pipeline_example_has_four_video_sources():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.example.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "wildtrack_gt_annotations"
    assert sorted(config["dataset"]["video_sources"].keys()) == ["C3", "C5", "C6", "C7"]


def test_single_source_sequential_offline_pipeline_config_is_present():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "single_source_sequential_replay"
    assert config["single_source_replay"]["source_camera_id"] == "C6"
    assert config["single_source_replay"]["virtual_camera_ids"] == ["C1", "C2", "C3", "C4"]
    assert config["face_demo_overrides"]["demo_auto_enroll_count"] == 0


def test_single_source_sequential_video_phase_config_is_present():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_3min.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "single_source_sequential_replay"
    assert config["low_load"]["start_frame"] == 0
    assert config["low_load"]["end_frame"] == 1800
    assert config["single_source_replay"]["virtual_time_offsets_sec"] == [0, 6, 12, 18]


def test_single_source_sequential_inference_phase_config_is_present():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_5min.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "single_source_sequential_replay"
    assert config["single_source_replay"]["track_provider"] == "inference_yolo_bytetrack"
    assert config["low_load"]["end_frame"] == 3000
    assert config["low_load"]["frame_stride"] == 3


def test_single_source_sequential_inference_90s_config_is_present():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_90s.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "single_source_sequential_replay"
    assert config["single_source_replay"]["track_provider"] == "inference_yolo_bytetrack"
    assert config["low_load"]["end_frame"] == 5394
    assert config["low_load"]["frame_stride"] == 6


def test_single_source_sequential_inference_50s_config_is_present():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_50s.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "single_source_sequential_replay"
    assert config["single_source_replay"]["track_provider"] == "inference_yolo_bytetrack"
    assert config["low_load"]["end_frame"] == 3000
    assert config["low_load"]["frame_stride"] == 6
    assert config["single_source_replay"]["inference"]["tracklet_linking"]["enabled"] is True


def test_single_source_sequential_inference_cache_benchmark_config_is_present():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_cache_benchmark.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "single_source_sequential_replay"
    assert config["low_load"]["end_frame"] == 720
    assert config["single_source_replay"]["inference"]["cache"]["enabled"] is True


def test_wildtrack_4cam_roi_benchmark_config_is_present():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.wildtrack_4cam_inference_roi_benchmark.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "wildtrack_video_inference"
    assert sorted(config["dataset"]["video_sources"].keys()) == ["C3", "C5", "C6", "C7"]
    assert config["multi_source_inference"]["roi_filter"]["enabled"] is True
    assert config["multi_source_inference"]["conf_threshold"] == 0.45
    assert config["multi_source_inference"]["iou_threshold"] == 0.35
    assert config["multi_source_inference"]["tracker"].endswith("bytetrack.wildtrack_4cam_strict.yaml")


def test_wildtrack_4cam_no_roi_benchmark_config_is_present():
    config_path = Path("insightface_demo_assets/runtime/config/offline_pipeline_demo.wildtrack_4cam_inference_no_roi_benchmark.yaml")
    config = load_pipeline_config(config_path)
    assert config["source_backend"] == "wildtrack_video_inference"
    assert sorted(config["dataset"]["video_sources"].keys()) == ["C3", "C5", "C6", "C7"]
    assert config["multi_source_inference"]["roi_filter"]["enabled"] is False
    assert config["multi_source_inference"]["conf_threshold"] == 0.45
    assert config["multi_source_inference"]["iou_threshold"] == 0.35
    assert config["multi_source_inference"]["tracker"].endswith("bytetrack.wildtrack_4cam_strict.yaml")


def test_processing_roi_filter_uses_bottom_center_footpoint():
    detections = [
        # Box center is inside the polygon, but the footpoint is below it and must be rejected.
        {"xmin": 40, "ymin": 20, "xmax": 80, "ymax": 120, "confidence": 0.9, "class_id": 0, "track_id": 1},
        # Box footpoint remains inside the polygon and must survive.
        {"xmin": 20, "ymin": 20, "xmax": 60, "ymax": 60, "confidence": 0.9, "class_id": 0, "track_id": 2},
    ]
    runtime_camera = {
        "processing_roi": [(0, 0), (100, 0), (100, 100), (0, 100)],
    }
    filtered, stats = filter_boxes_by_processing_roi(
        detections,
        runtime_camera,
        require_footpoint_inside=True,
        min_bbox_coverage=0.0,
    )
    kept_ids = {item["track_id"] for item in filtered}
    assert kept_ids == {2}
    assert stats["killed_count"] == 1
    assert stats["killed_footpoint_count"] == 1


def test_tracklet_linking_reconnects_short_occlusion():
    rows = [
        _track_row(10, 0, 10, 10, 50, 110),
        _track_row(10, 1, 20, 10, 60, 110),
        _track_row(10, 2, 30, 10, 70, 110),
        _track_row(11, 6, 70, 10, 110, 110),
        _track_row(11, 7, 80, 10, 120, 110),
        _track_row(30, 0, 180, 10, 220, 110),
        _track_row(30, 1, 182, 10, 222, 110),
    ]
    linked_rows, runtime = link_short_gap_tracklets(
        rows,
        {
            "enabled": True,
            "max_gap_frames": 8,
            "min_iou": 0.05,
            "max_normalized_center_distance": 0.8,
            "min_motion_continuity": 0.1,
        },
    )
    linked_ids = {row["local_track_id"] for row in linked_rows}
    assert runtime["original_tracklet_count"] == 3
    assert runtime["linked_tracklet_count"] == 2
    assert runtime["links_applied"] == 1
    assert "10" in linked_ids
    assert not any(row["local_track_id"] == "11" for row in linked_rows)


def test_tracklet_linking_handles_crossing_without_false_merge():
    rows = [
        _track_row(1, 0, 10, 20, 50, 120),
        _track_row(1, 1, 20, 20, 60, 120),
        _track_row(1, 2, 30, 20, 70, 120),
        _track_row(2, 0, 120, 20, 160, 120),
        _track_row(2, 1, 110, 20, 150, 120),
        _track_row(2, 2, 100, 20, 140, 120),
        _track_row(2, 3, 90, 20, 130, 120),
        _track_row(3, 4, 50, 20, 90, 120),
        _track_row(3, 5, 60, 20, 100, 120),
    ]
    linked_rows, runtime = link_short_gap_tracklets(
        rows,
        {
            "enabled": True,
            "max_gap_frames": 6,
            "min_iou": 0.05,
            "max_normalized_center_distance": 0.75,
            "min_motion_continuity": 0.1,
        },
    )
    remapped_rows = [row for row in linked_rows if row.get("bytetrack_track_id") == "3"]
    assert runtime["links_applied"] == 1
    assert remapped_rows
    assert {row["local_track_id"] for row in remapped_rows} == {"1"}
    assert any(row["local_track_id"] == "2" for row in linked_rows)


def test_tracklet_linking_keeps_close_parallel_people_separate_when_motion_breaks():
    rows = [
        _track_row(5, 0, 10, 15, 50, 115),
        _track_row(5, 1, 20, 15, 60, 115),
        _track_row(6, 0, 55, 15, 95, 115),
        _track_row(6, 1, 65, 15, 105, 115),
        _track_row(7, 4, 150, 15, 190, 115),
        _track_row(7, 5, 160, 15, 200, 115),
    ]
    linked_rows, runtime = link_short_gap_tracklets(
        rows,
        {
            "enabled": True,
            "max_gap_frames": 6,
            "min_iou": 0.1,
            "max_normalized_center_distance": 0.55,
            "min_motion_continuity": 0.2,
        },
    )
    assert runtime["links_applied"] == 0
    assert {row["local_track_id"] for row in linked_rows} == {"5", "6", "7"}


def test_inference_track_cache_reports_miss_then_hit(tmp_path):
    cache_dir = tmp_path / "cache"
    signature = {
        "camera_id": "C6",
        "video_source": "cam6.mp4",
        "start_frame": 0,
        "end_frame": 100,
        "frame_stride": 6,
        "detector_model": "yolov8n.pt",
        "tracker": "bytetrack.yaml",
        "device": "cpu",
        "conf_threshold": 0.25,
        "iou_threshold": 0.5,
        "imgsz": 640,
        "person_class_id": 0,
        "target_timeline_fps": 10.0,
        "actual_video_fps": 59.94,
        "tracklet_linking": {"enabled": True},
    }
    rows, miss_runtime = load_inference_track_cache(cache_dir, signature)
    assert rows is None
    assert miss_runtime["cache_hit"] is False

    track_rows = [_track_row(10, 0, 10, 10, 50, 110)]
    write_inference_track_cache(cache_dir, signature, track_rows, {"elapsed_sec": 12.5})

    cached_rows, hit_runtime = load_inference_track_cache(cache_dir, signature)
    assert hit_runtime["cache_hit"] is True
    assert len(cached_rows) == 1
    assert cached_rows[0]["local_track_id"] == "10"


def test_inference_track_cache_signature_mismatch_invalidates_hit(tmp_path):
    cache_dir = tmp_path / "cache"
    signature = {
        "camera_id": "C6",
        "video_source": "cam6.mp4",
        "start_frame": 0,
        "end_frame": 100,
        "frame_stride": 6,
        "detector_model": "yolov8n.pt",
        "tracker": "bytetrack.yaml",
        "device": "cpu",
        "conf_threshold": 0.25,
        "iou_threshold": 0.5,
        "imgsz": 640,
        "person_class_id": 0,
        "target_timeline_fps": 10.0,
        "actual_video_fps": 59.94,
        "tracklet_linking": {"enabled": True},
    }
    write_inference_track_cache(cache_dir, signature, [_track_row(10, 0, 10, 10, 50, 110)], {"elapsed_sec": 8.0})
    mismatched_signature = dict(signature)
    mismatched_signature["frame_stride"] = 3
    rows, runtime = load_inference_track_cache(cache_dir, mismatched_signature)
    assert rows is None
    assert runtime["cache_hit"] is False


def test_clone_track_rows_for_virtual_camera_applies_offsets():
    source_rows = [
        {
            "camera_id": "C6",
            "role": "entry",
            "local_track_id": "7",
            "global_gt_id": 7,
            "frame_id": 120,
            "frame_key": "00000120",
            "relative_sec": 12.0,
            "video_path": "cam6.mp4",
            "image_rel_path": "Image_subsets\\C6\\00000120.png",
        }
    ]
    cloned = clone_track_rows_for_virtual_camera(source_rows, "C2", "entry", 6.0, 60)
    assert len(cloned) == 1
    assert cloned[0]["camera_id"] == "C2"
    assert cloned[0]["source_camera_id"] == "C6"
    assert cloned[0]["source_local_track_id"] == "7"
    assert cloned[0]["local_track_id"] == "C2_7"
    assert cloned[0]["frame_id"] == 180
    assert cloned[0]["relative_sec"] == 18.0
    assert cloned[0]["fake_time_offset_sec"] == 6.0
    assert cloned[0]["fake_frame_offset"] == 60


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
