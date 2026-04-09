from run_association_tuning import pairwise_unknown_metrics, split_merge_metrics


def test_pairwise_unknown_metrics_counts_tp_fp_fn():
    rows = [
        {"identity_status": "unknown", "camera_id": "C1", "global_gt_id": "1", "unknown_global_id": "UNK_1"},
        {"identity_status": "unknown", "camera_id": "C2", "global_gt_id": "1", "unknown_global_id": "UNK_1"},
        {"identity_status": "unknown", "camera_id": "C3", "global_gt_id": "2", "unknown_global_id": "UNK_1"},
        {"identity_status": "unknown", "camera_id": "C4", "global_gt_id": "2", "unknown_global_id": "UNK_2"},
    ]
    metrics = pairwise_unknown_metrics(rows)
    assert metrics["pairwise_tp"] == 1
    assert metrics["pairwise_fp"] >= 1
    assert metrics["pairwise_fn"] >= 1


def test_split_merge_metrics_detects_merge_and_split():
    rows = [
        {"identity_status": "unknown", "camera_id": "C1", "global_gt_id": "1", "unknown_global_id": "UNK_1"},
        {"identity_status": "unknown", "camera_id": "C2", "global_gt_id": "2", "unknown_global_id": "UNK_1"},
        {"identity_status": "unknown", "camera_id": "C3", "global_gt_id": "3", "unknown_global_id": "UNK_2"},
        {"identity_status": "unknown", "camera_id": "C4", "global_gt_id": "3", "unknown_global_id": "UNK_3"},
    ]
    metrics = split_merge_metrics(rows)
    assert metrics["merge_error_count"] == 1
    assert metrics["split_gt_count"] == 1
