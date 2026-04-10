from pathlib import Path

import pytest

from offline_pipeline.direction_logic import evaluate_direction
from scene_calibration import (
    apply_scene_calibration_to_transition_map,
    apply_scene_calibration_to_wildtrack_config,
    load_runtime_scene_calibration,
)


def _sample_calibration_payload():
    return {
        "scene_calibration": {
            "coordinate_space": "normalized",
            "processing": {"require_manual_calibration": True, "roi_mask_enabled": True},
            "direction_filter": {
                "history_window": 4,
                "minimum_points": 4,
                "minimum_inward_motion_px": 8.0,
                "minimum_inside_ratio": 0.5,
                "require_zone_transition": False,
                "allow_line_cross_only_when_history_short": False,
                "inward_zone_types": ["exit", "interior"],
                "outward_zone_types": ["entry", "outer"],
            },
            "cameras": {
                "C5": {
                    "camera_id": "C5",
                    "role": "entry",
                    "preview_source": "",
                    "preview_source_type": "file",
                    "frame_size_ref": {"width": 100, "height": 100},
                    "processing_roi": {"polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]},
                    "entry_line": {"points": [[0.0, 0.5], [1.0, 0.5]], "in_side_point": [0.5, 0.8]},
                    "default_zone_id": "c5_entry_main",
                    "default_subzone_id": "c5_inner_exit",
                    "entry_zones": ["c5_entry_main"],
                    "exit_zones": ["c5_entry_main"],
                    "zones": [
                        {
                            "zone_id": "c5_entry_main",
                            "zone_type": "entry",
                            "priority": 10,
                            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        }
                    ],
                    "subzones": [
                        {
                            "subzone_id": "c5_outer_entry",
                            "parent_zone_id": "c5_entry_main",
                            "subzone_type": "entry",
                            "priority": 100,
                            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.45], [0.0, 0.45]],
                        },
                        {
                            "subzone_id": "c5_inner_exit",
                            "parent_zone_id": "c5_entry_main",
                            "subzone_type": "exit",
                            "priority": 110,
                            "polygon": [[0.0, 0.45], [1.0, 0.45], [1.0, 1.0], [0.0, 1.0]],
                        },
                    ],
                }
            },
        }
    }


def test_manual_scene_calibration_is_required_for_runtime(tmp_path):
    invalid_path = tmp_path / "invalid_scene.json"
    invalid_path.write_text('{"scene_calibration":{"coordinate_space":"normalized","cameras":{"C5":{"processing_roi":{"polygon":[]}}}}}', encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_runtime_scene_calibration(
            config_path=str(invalid_path),
            base_dir=tmp_path,
            camera_ids=["C5"],
            required=True,
        )


def test_runtime_scene_calibration_uses_repo_default_when_requested_path_missing(tmp_path):
    calibration, runtime_cameras, runtime = load_runtime_scene_calibration(
        config_path=str(tmp_path / "missing.json"),
        base_dir=tmp_path,
        camera_ids=["C5"],
        required=False,
    )
    assert runtime["preview_only"] is False
    assert "C5" in calibration["cameras"]
    assert "C5" in runtime_cameras


def test_scene_calibration_merges_geometry_into_transition_map_and_wildtrack(tmp_path):
    calibration_path = tmp_path / "manual_scene.json"
    calibration_path.write_text(__import__("json").dumps(_sample_calibration_payload()), encoding="utf-8")
    calibration, _runtime_cameras, runtime = load_runtime_scene_calibration(
        config_path=str(calibration_path),
        base_dir=tmp_path,
        camera_ids=["C5"],
        required=True,
    )
    frame_sizes = runtime["frame_sizes"]
    transition_map = {
        "cameras": {"C5": {"default_zone_id": "legacy_zone", "zones": [], "subzones": []}},
        "transitions": [],
    }
    merged_map = apply_scene_calibration_to_transition_map(transition_map, calibration, frame_sizes)
    assert merged_map["runtime_policy"]["disallow_default_region_fallback"] is True
    assert merged_map["cameras"]["C5"]["processing_roi"]
    assert merged_map["cameras"]["C5"]["manual_calibration_active"] is True

    wildtrack_config = {
        "selected_cameras": ["C5"],
        "cameras": {"C5": {"role": "entry"}},
    }
    merged_wildtrack = apply_scene_calibration_to_wildtrack_config(wildtrack_config, calibration, frame_sizes)
    assert merged_wildtrack["scene_calibration_required"] is True
    assert merged_wildtrack["cameras"]["C5"]["entry_line"]
    assert merged_wildtrack["cameras"]["C5"]["processing_roi"]
    assert merged_wildtrack["direction_filter"]["history_window"] == 4


def test_direction_logic_requires_history_and_momentum():
    line = [[0, 50], [100, 50]]
    in_side_point = [50, 80]
    weak_points = [
        {"x": 50, "y": 47},
        {"x": 50, "y": 49},
        {"x": 50, "y": 50.5},
        {"x": 50, "y": 52},
    ]
    strong_points = [
        {"x": 50, "y": 10},
        {"x": 50, "y": 30},
        {"x": 50, "y": 55},
        {"x": 50, "y": 85},
    ]
    config = {
        "history_window": 4,
        "minimum_points": 4,
        "minimum_inward_motion_px": 8.0,
        "minimum_inside_ratio": 0.5,
        "require_zone_transition": False,
        "allow_line_cross_only_when_history_short": False,
        "inward_zone_types": ["exit", "interior"],
        "outward_zone_types": ["entry", "outer"],
    }
    weak_result = evaluate_direction(weak_points, line, in_side_point, spatial_history=[], config=config)
    strong_result = evaluate_direction(strong_points, line, in_side_point, spatial_history=[], config=config)
    assert weak_result["decision"] == "NONE"
    assert strong_result["decision"] == "IN"
