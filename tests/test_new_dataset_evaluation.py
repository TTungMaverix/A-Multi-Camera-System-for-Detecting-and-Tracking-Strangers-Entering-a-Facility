from pathlib import Path

import json

from run_new_dataset_evaluation import build_regression_summary, build_temp_pipeline_config, summarize_appearance_vs_topology
from run_new_dataset_inventory import calibration_reuse_assessment
from run_new_dataset_pair_debug import build_root_cause_summary


def test_build_temp_pipeline_config_sets_pair_and_disables_cache(tmp_path):
    base_config = {
        "pipeline_name": "base_demo",
        "logical_demo": {"pair_id": ""},
        "multi_source_inference": {
            "cache": {
                "enabled": True,
                "use_cache": True,
                "refresh_cache": False,
            }
        },
    }
    payload = build_temp_pipeline_config(base_config, "a2", tmp_path / "run_a2", tmp_path)
    offline_config = payload["offline_pipeline"]
    assert offline_config["pipeline_name"] == "new_dataset_eval_a2"
    assert offline_config["logical_demo"]["pair_id"] == "a2"
    assert offline_config["multi_source_inference"]["cache"]["enabled"] is False
    assert offline_config["multi_source_inference"]["cache"]["use_cache"] is False


def test_summarize_appearance_vs_topology_counts_rescue_and_threshold_pass():
    decision_logs = [
        {
            "decision": "unknown_reuse",
            "candidate_evaluations": [
                {
                    "candidate_unknown_global_id": "UNK_0001",
                    "topology_allowed": True,
                    "body_score": 0.81,
                    "appearance_primary": 0.81,
                    "appearance_secondary": 0.0,
                    "acceptance_reason": "candidate_threshold_pass",
                    "topology_support_level": "strong",
                    "relation_type": "sequential",
                }
            ],
            "thresholds_used": {"primary_threshold": 0.72},
        },
        {
            "decision": "unknown_reuse",
            "candidate_evaluations": [
                {
                    "candidate_unknown_global_id": "UNK_0001",
                    "topology_allowed": True,
                    "body_score": 0.66,
                    "appearance_primary": 0.66,
                    "appearance_secondary": 0.0,
                    "acceptance_reason": "topology_supported_body_accept",
                    "topology_support_level": "strong",
                    "relation_type": "sequential",
                }
            ],
            "thresholds_used": {"primary_threshold": 0.72},
        },
        {
            "decision": "create_new",
            "reason_code": "below_primary_threshold",
            "candidate_evaluations": [
                {
                    "candidate_unknown_global_id": "UNK_0002",
                    "topology_allowed": True,
                    "body_score": 0.61,
                    "appearance_primary": 0.61,
                    "appearance_secondary": 0.0,
                    "acceptance_reason": "below_primary_threshold",
                    "topology_support_level": "moderate",
                    "relation_type": "sequential",
                }
            ],
            "thresholds_used": {"primary_threshold": 0.72},
        },
    ]

    summary = summarize_appearance_vs_topology(decision_logs)

    assert summary["unknown_reuse_count"] == 2
    assert summary["appearance_only_pass_count"] == 1
    assert summary["topology_supported_pass_count"] == 1
    assert summary["topology_rescued_count"] == 1
    assert summary["appearance_only_fail_count"] == 1


def test_calibration_reuse_assessment_accepts_same_height_small_aspect_delta():
    dataset_profile = {
        "selected_cameras": ["C1", "C2"],
        "cameras": {
            "C1": {
                "source_physical_camera_id": "CAM1",
                "logical_demo_copy": False,
            }
        },
    }
    calibration = {
        "cameras": {
            "C1": {
                "anchor_point_mode": "bottom_center",
                "frame_size_ref": {"width": 1418, "height": 720},
            }
        }
    }
    result = calibration_reuse_assessment(
        dataset_profile,
        calibration,
        "CAM1",
        {"width": 1404, "height": 720},
        max_aspect_ratio_delta=0.12,
    )
    assert result["reusable"] is True
    assert result["calibration_camera_id"] == "C1"
    assert result["anchor_point_mode"] == "bottom_center"


def test_build_regression_summary_compares_before_after(tmp_path):
    baseline_dir = tmp_path / "baseline"
    per_clip_dir = baseline_dir / "per_clip_evaluation"
    per_clip_dir.mkdir(parents=True)
    baseline_payload = {
        "pair_id": "a3",
        "identity_summary": {"total_event_count": 3, "unknown_reuse_count": 0},
        "appearance_vs_topology": {
            "topology_rescued_count": 0,
            "records": [{"appearance_only_body_score": 0.5071}],
        },
    }
    (per_clip_dir / "a3.json").write_text(json.dumps(baseline_payload), encoding="utf-8")
    current_rows = [
        {
            "pair_id": "a3",
            "identity_summary": {"total_event_count": 4, "unknown_reuse_count": 1},
            "appearance_vs_topology": {
                "topology_rescued_count": 1,
                "records": [{"appearance_only_body_score": 0.58}],
            },
        }
    ]

    summary = build_regression_summary(current_rows, baseline_output_dir=baseline_dir)

    assert summary["per_clip"]["a3"]["before"]["total_event_count"] == 3
    assert summary["per_clip"]["a3"]["after"]["total_event_count"] == 4
    assert summary["per_clip"]["a3"]["delta"]["unknown_reuse_count"] == 1.0
    assert summary["per_clip"]["a3"]["delta"]["appearance_only_max_body_score"] == 0.0729


def test_build_root_cause_summary_reads_detector_runtime_from_physical_camera_payload():
    stage_input_summary = {
        "multi_source_inference": {
            "per_physical_camera_runtime": {
                "CAM1": {"raw_detection_count": 6},
                "CAM2": {"raw_detection_count": 0},
            }
        }
    }
    track_debug = {
        "C1": {"track_count": 1, "tracks": []},
        "C2": {"track_count": 0, "tracks": []},
    }
    missing_event_rows = []
    entry_events = [{"event_id": "IN_C1_001", "direction_accept_mode": "late_start_inside_entry"}]

    summary = build_root_cause_summary(stage_input_summary, track_debug, missing_event_rows, entry_events)

    assert summary["stage_counts"]["detector_alive"] is True
    assert summary["stage_counts"]["tracker_alive"] is True
    assert summary["stage_counts"]["entry_event_count"] == 1
    assert any("late-start inside-entry fallback" in line for line in summary["root_cause_lines"])
