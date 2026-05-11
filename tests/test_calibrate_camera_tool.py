import json
from argparse import Namespace

import numpy as np
import yaml

from scene_calibration import load_scene_calibration
from tools.calibrate_camera import (
    build_camera_config,
    classify_movement_by_direction_vector,
    compute_in_direction_vector,
    write_outputs,
)


def test_compute_in_direction_vector_points_to_clicked_side():
    p1 = [0, 50]
    p2 = [100, 50]
    in_side = [50, 90]

    vector = compute_in_direction_vector(p1, p2, in_side)
    decision, score = classify_movement_by_direction_vector([50, 40], [50, 80], vector)

    assert vector == [0.0, 1.0]
    assert decision == "IN"
    assert score > 0


def test_classify_movement_by_direction_vector_marks_outward_motion():
    direction, score = classify_movement_by_direction_vector([10, 20], [10, 5], [0, 1])

    assert direction == "OUT"
    assert score < 0


def test_build_camera_config_matches_runtime_schema():
    cfg = build_camera_config(
        camera_id="C9",
        source="demo.mp4",
        source_type="file",
        width=200,
        height=100,
        roi_points=[[10, 10], [190, 10], [190, 90], [10, 90]],
        entry_points=[[30, 50], [170, 50], [100, 80]],
        anchor_point_mode="center_center",
    )

    assert cfg["processing_roi"]["polygon"][0] == [0.05, 0.1]
    assert cfg["entry_line"]["points"] == [[0.15, 0.5], [0.85, 0.5]]
    assert cfg["entry_line"]["in_side_point"] == [0.5, 0.8]
    assert cfg["default_zone_id"] == "c9_entry_main"
    assert cfg["zones"][0]["polygon"] == cfg["processing_roi"]["polygon"]


def test_write_outputs_merges_yaml_and_per_camera_json(tmp_path):
    output_config = tmp_path / "manual_scene_calibration.custom.yaml"
    camera_json = tmp_path / "C9.json"
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    args = Namespace(
        camera="C9",
        source="demo.mp4",
        output_config=str(output_config),
        base_config="",
        per_camera_json=str(camera_json),
        anchor_point_mode="bottom_center",
        role="entry",
        description="test camera",
        no_default_zone=False,
    )

    write_outputs(
        args,
        frame,
        "file",
        [[10, 10], [190, 10], [190, 90], [10, 90]],
        [[30, 50], [170, 50], [100, 80]],
    )

    payload = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    assert "scene_calibration" in payload
    calibration, runtime = load_scene_calibration(str(output_config), base_dir=tmp_path, required=True)
    assert runtime["errors"] == []
    assert "C9" in calibration["cameras"]

    per_camera = json.loads(camera_json.read_text(encoding="utf-8"))
    assert per_camera["camera_id"] == "C9"
    assert per_camera["resolution"] == [200, 100]
    assert per_camera["entry_line"]["in_direction_vector"] == [0.0, 1.0]
