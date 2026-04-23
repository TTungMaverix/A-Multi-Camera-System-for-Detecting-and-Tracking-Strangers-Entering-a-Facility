from pathlib import Path

from run_new_dataset_evaluation import build_temp_pipeline_config, summarize_appearance_vs_topology
from run_new_dataset_inventory import calibration_reuse_assessment


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
