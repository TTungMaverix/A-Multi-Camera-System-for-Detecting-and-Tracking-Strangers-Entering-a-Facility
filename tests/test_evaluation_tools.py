import json

from evaluation_utils import (
    build_unknown_timeline,
    compute_event_level_idf1,
    match_event_rows_to_gt,
    summarize_latency_records,
)
from run_threshold_analysis import choose_recommended_threshold


def test_match_event_rows_to_gt_uses_frame_and_foot_proximity():
    gt_rows = [
        {"event_id": "GT1", "camera_id": "C1", "frame_id": "100", "foot_x": "400", "foot_y": "600", "global_gt_id": "11"},
        {"event_id": "GT2", "camera_id": "C1", "frame_id": "120", "foot_x": "800", "foot_y": "610", "global_gt_id": "40"},
    ]
    pred_rows = [
        {"event_id": "P1", "camera_id": "C1", "frame_id": "101", "foot_x": "402", "foot_y": "602", "resolved_global_id": "UNK_0001"},
        {"event_id": "P2", "camera_id": "C1", "frame_id": "119", "foot_x": "796", "foot_y": "608", "resolved_global_id": "UNK_0002"},
    ]
    matches, unmatched_gt, unmatched_pred = match_event_rows_to_gt(gt_rows, pred_rows, frame_tolerance=3, max_foot_distance_px=20.0)
    assert len(matches) == 2
    assert not unmatched_gt
    assert not unmatched_pred
    assert {row["gt_global_id"] for row in matches} == {"11", "40"}


def test_match_event_rows_to_gt_prefers_anchor_buffer_coordinates():
    gt_rows = [
        {
            "event_id": "GT1",
            "camera_id": "C1",
            "frame_id": "100",
            "foot_x": "900",
            "foot_y": "900",
            "global_gt_id": "11",
            "evidence_buffer_json": json.dumps([{"frame_id": 100, "foot_x": 400, "foot_y": 600}]),
        }
    ]
    pred_rows = [
        {
            "event_id": "P1",
            "camera_id": "C1",
            "frame_id": "101",
            "foot_x": "910",
            "foot_y": "910",
            "resolved_global_id": "UNK_0001",
            "evidence_buffer_json": json.dumps([{"frame_id": 101, "foot_x": 404, "foot_y": 603}]),
        }
    ]
    matches, unmatched_gt, unmatched_pred = match_event_rows_to_gt(gt_rows, pred_rows, frame_tolerance=3, max_foot_distance_px=10.0)
    assert len(matches) == 1
    assert not unmatched_gt
    assert not unmatched_pred


def test_compute_event_level_idf1_is_perfect_when_ids_are_consistent():
    gt_rows = [
        {"global_gt_id": "11"},
        {"global_gt_id": "11"},
        {"global_gt_id": "40"},
        {"global_gt_id": "40"},
    ]
    pred_rows = [
        {"resolved_global_id": "UNK_0001"},
        {"resolved_global_id": "UNK_0001"},
        {"resolved_global_id": "UNK_0002"},
        {"resolved_global_id": "UNK_0002"},
    ]
    matches = [
        {"gt_global_id": "11", "pred_identity_id": "UNK_0001"},
        {"gt_global_id": "11", "pred_identity_id": "UNK_0001"},
        {"gt_global_id": "40", "pred_identity_id": "UNK_0002"},
        {"gt_global_id": "40", "pred_identity_id": "UNK_0002"},
    ]
    metrics = compute_event_level_idf1(gt_rows, pred_rows, matches)
    assert metrics["idf1"] == 1.0
    assert metrics["idtp"] == 4
    assert metrics["idfp"] == 0
    assert metrics["idfn"] == 0


def test_build_unknown_timeline_groups_sorted_appearances():
    resolved_rows = [
        {
            "event_id": "E2",
            "identity_status": "unknown",
            "resolved_global_id": "UNK_0001",
            "ui_identity_label": "UNK_0001",
            "camera_id": "C2",
            "relative_sec": "9.1",
            "relation_type": "sequential",
            "zone_id": "c2_entry",
            "subzone_id": "c2_inner",
            "modality_primary_used": "fusion",
            "modality_secondary_used": "face_body_context",
            "decision_reason": "unknown_reuse",
            "reason_code": "unknown_reuse",
            "best_body_crop": "body2.png",
            "best_head_crop": "head2.png",
        },
        {
            "event_id": "E1",
            "identity_status": "unknown",
            "resolved_global_id": "UNK_0001",
            "ui_identity_label": "UNK_0001",
            "camera_id": "C1",
            "relative_sec": "3.1",
            "relation_type": "entry",
            "zone_id": "c1_entry",
            "subzone_id": "c1_inner",
            "modality_primary_used": "face",
            "modality_secondary_used": "",
            "decision_reason": "create_new",
            "reason_code": "create_new",
            "best_body_crop": "body1.png",
            "best_head_crop": "head1.png",
        },
    ]
    rows, payload = build_unknown_timeline(resolved_rows)
    assert len(rows) == 2
    assert len(payload) == 1
    assert payload[0]["identity_id"] == "UNK_0001"
    assert payload[0]["camera_sequence"] == ["C1", "C2"]
    assert payload[0]["appearances"][0]["event_id"] == "E1"


def test_summarize_latency_records_returns_p95():
    summary = summarize_latency_records([0.5, 0.9, 1.1, 1.3, 2.0])
    assert summary["count"] == 5
    assert summary["avg_latency_sec"] > 0.0
    assert summary["p95_latency_sec"] >= summary["p50_latency_sec"]


def test_choose_recommended_threshold_keeps_current_value_inside_safe_gap():
    metrics = {
        "positive_count": 6,
        "negative_count": 3,
        "positive_min": 1.0,
        "negative_max": 0.2361,
        "best_f1_threshold": {"threshold": 0.2361},
    }
    assert choose_recommended_threshold(metrics, 0.618) == 0.618
