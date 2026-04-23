from pathlib import Path

from association_core import load_camera_transition_map
from dataset_profiles import (
    build_logical_demo_manifest,
    discover_clip_pairs,
    load_dataset_profile,
    materialize_profile_for_logical_demo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "insightface_demo_assets" / "runtime" / "config" / "dataset_profile.new_dataset_demo.yaml"
TRANSITION_PATH = PROJECT_ROOT / "insightface_demo_assets" / "runtime" / "config" / "camera_transition_map.new_dataset_demo.yaml"


def test_new_dataset_profile_loads_and_marks_anchor_modes():
    profile, runtime = load_dataset_profile(PROFILE_PATH)
    assert runtime["dataset_name"] == "New Dataset"
    assert profile["selected_cameras"] == ["C1", "C2", "C3", "C4"]
    assert profile["cameras"]["C1"]["anchor_point_mode"] == "bottom_center"
    assert profile["cameras"]["C2"]["anchor_point_mode"] == "center_center"
    assert profile["logical_demo"]["enabled"] is True


def test_new_dataset_adapter_discovers_shared_stem_pairs():
    profile, _runtime = load_dataset_profile(PROFILE_PATH)
    discovered = discover_clip_pairs(profile, PROJECT_ROOT)
    assert "a1" in discovered["available_pair_ids"]
    assert discovered["files_by_camera"]["CAM1"]["a1"].name == "a1.mp4"
    assert discovered["files_by_camera"]["CAM2"]["a1"].name == "a1.mp4"


def test_new_dataset_logical_demo_manifest_expands_two_physical_cameras_to_four_logical():
    profile, _runtime = load_dataset_profile(PROFILE_PATH)
    manifest = build_logical_demo_manifest(profile, PROJECT_ROOT, requested_pair_id="a1")
    logical_ids = [row["logical_camera_id"] for row in manifest["logical_cameras"]]
    offsets = [row["timeline_offset_sec"] for row in manifest["logical_cameras"]]
    assert logical_ids == ["C1", "C2", "C3", "C4"]
    assert offsets == [0.0, 10.0, 20.0, 30.0]
    assert len(manifest["physical_sources"]) == 2


def test_materialized_logical_profile_keeps_pair_video_sources():
    profile, _runtime = load_dataset_profile(PROFILE_PATH)
    manifest = build_logical_demo_manifest(profile, PROJECT_ROOT, requested_pair_id="a1")
    materialized = materialize_profile_for_logical_demo(profile, manifest)
    assert materialized["cameras"]["C1"]["preview_source"].endswith("Camera 1\\a1.mp4")
    assert materialized["cameras"]["C2"]["preview_source"].endswith("Camera 2\\a1.mp4")
    assert materialized["cameras"]["C3"]["logical_demo_copy"] is True
    assert materialized["cameras"]["C4"]["timeline_offset_sec"] == 30.0


def test_new_dataset_transition_map_loads_physical_and_logical_windows():
    profile, _runtime = load_dataset_profile(PROFILE_PATH)
    transition_map, runtime = load_camera_transition_map(profile, config_path=str(TRANSITION_PATH), base_dir=PROJECT_ROOT)
    assert runtime["camera_count"] == 4
    assert runtime["transition_count"] == 3
    by_id = {item["transition_id"]: item for item in transition_map["transitions"]}
    assert by_id["C1_to_C2_physical"]["min_travel_time_sec"] == 5.0
    assert by_id["C1_to_C2_physical"]["max_travel_time_sec"] == 30.0
    assert by_id["C2_to_C3_demo"]["allowed_entry_subzones"] == ["c3_outer_entry", "c3_inner_entry"]
    assert by_id["C3_to_C4_demo"]["allowed_exit_subzones"] == ["c3_inner_entry"]
