import numpy as np

from association_core.config_loader import load_association_policy
from association_core.decision_policy import assign_model_identities
from association_core.gallery_lifecycle import create_unknown_profile, expire_profiles, update_unknown_profile
from association_core.spatial_context import build_event_assignment_audit_row, resolve_spatial_context
from association_core.topology_filter import build_topology_index


def make_topology(relation, min_sec=0.0, max_sec=1.0, avg_sec=None):
    fps = 10.0
    avg = avg_sec if avg_sec is not None else (min_sec + max_sec) / 2.0
    min_gap = int(round(min_sec * fps))
    avg_gap = int(round(avg * fps))
    max_gap = int(round(max_sec * fps))
    return build_topology_index(
        {
            "assumed_video_fps": fps,
            "camera_topology": {
                "C1": {
                    "C2": {
                        "relation": relation,
                        "min_frame_gap": min_gap,
                        "max_frame_gap": max_gap,
                    }
                },
                "C3": {
                    "C2": {
                        "relation": relation,
                        "min_frame_gap": min_gap,
                        "max_frame_gap": max_gap,
                    }
                },
            },
        }
    )


def make_transition_topology(
    relation,
    src_camera="C1",
    dst_camera="C2",
    min_sec=0.0,
    max_sec=1.0,
    avg_sec=None,
    allowed_exit_zones=None,
    allowed_entry_zones=None,
    allowed_exit_subzones=None,
    allowed_entry_subzones=None,
):
    avg = avg_sec if avg_sec is not None else (min_sec + max_sec) / 2.0
    return build_topology_index(
        {
            "cameras": {
                src_camera: {
                    "default_zone_id": allowed_exit_zones[0] if allowed_exit_zones else "",
                    "default_subzone_id": allowed_exit_subzones[0] if allowed_exit_subzones else "",
                },
                dst_camera: {
                    "default_zone_id": allowed_entry_zones[0] if allowed_entry_zones else "",
                    "default_subzone_id": allowed_entry_subzones[0] if allowed_entry_subzones else "",
                },
            },
            "transitions": [
                {
                    "transition_id": f"{src_camera}_to_{dst_camera}_{relation}",
                    "src_camera_id": src_camera,
                    "dst_camera_id": dst_camera,
                    "relation_type": relation,
                    "allowed_relation_types": [relation],
                    "same_area_overlap": relation == "overlap",
                    "min_travel_time_sec": min_sec,
                    "avg_travel_time_sec": avg,
                    "max_travel_time_sec": max_sec,
                    "allowed_exit_zones": allowed_exit_zones or [],
                    "allowed_entry_zones": allowed_entry_zones or [],
                    "allowed_exit_subzones": allowed_exit_subzones or [],
                    "allowed_entry_subzones": allowed_entry_subzones or [],
                }
            ],
        }
    )


def make_item(
    event_id,
    camera_id,
    sec,
    face=None,
    body=None,
    bbox_area=4000,
    gt_id="1",
    zone_id="",
    subzone_id="",
    face_det_score=None,
):
    return {
        "event": {
            "event_id": event_id,
            "event_type": "FOLLOWUP_OBSERVATION",
            "camera_id": camera_id,
            "frame_id": int(round(sec * 10)),
            "relative_sec": float(sec),
            "global_gt_id": str(gt_id),
            "best_head_crop": "",
            "best_body_crop": "",
            "bbox_area": float(bbox_area),
            "anchor_camera_id": "",
            "anchor_relative_sec": "",
            "relation_type": "",
            "zone_id": zone_id,
            "subzone_id": subzone_id,
        },
        "face_embedding": np.asarray(face, dtype=np.float32) if face is not None else None,
        "face_status": "ok" if face is not None else "missing",
        "face_count": 1 if face is not None else 0,
        "face_det_score": (0.9 if face is not None else 0.0) if face_det_score is None else float(face_det_score),
        "used_face_crop": "head" if face is not None else "",
        "used_face_crop_path": "",
        "face_bbox": "",
        "body_embedding": np.asarray(body, dtype=np.float32) if body is not None else None,
        "body_status": "ok" if body is not None else "missing",
        "body_shape": "64x128" if body is not None else "",
    }


def default_policy():
    policy, _runtime = load_association_policy()
    return policy


def test_overlap_relation_reuses_same_unknown_id():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0]),
        make_item("e2", "C2", 0.2, face=[0.99, 0.01], body=[0.99, 0.01]),
    ]
    rows, profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("overlap", min_sec=0.0, max_sec=1.0),
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] == rows[1]["unknown_global_id"]
    assert rows[1]["resolution_source"] == "model_unknown_gallery_reuse"
    assert debug["decision_logs"][1]["decision"] == "unknown_reuse"


def test_sequential_relation_reuses_within_travel_window():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0]),
        make_item("e2", "C2", 2.0, face=[1.0, 0.0], body=[1.0, 0.0]),
    ]
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("sequential", min_sec=1.0, max_sec=3.0, avg_sec=2.0),
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] == rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["relation_type"] == "sequential"


def test_out_of_time_window_does_not_reuse():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0]),
        make_item("e2", "C2", 5.0, face=[1.0, 0.0], body=[1.0, 0.0]),
    ]
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("sequential", min_sec=1.0, max_sec=3.0, avg_sec=2.0),
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] != rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["reason_code"] in {"no_candidate", "no_previous_unknown_profiles", "sequential_window_reject"}


def test_camera_without_topology_link_does_not_reuse():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0]),
        make_item("e2", "C2", 0.2, face=[1.0, 0.0], body=[1.0, 0.0]),
    ]
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        {},
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] != rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["candidate_set_after_filter"] == []


def test_quality_gate_fail_defer():
    items = [make_item("e1", "C1", 0.0, face=None, body=None, bbox_area=200)]
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        {},
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["identity_status"] == "deferred"
    assert debug["decision_logs"][0]["decision"] == "defer"


def test_ambiguous_margin_enters_pending_instead_of_creating_new_immediately():
    items = [
        make_item("e1", "C1", 0.0, body=[1.0, 0.0]),
        make_item("e2", "C3", 0.1, body=[0.0, 1.0], gt_id="2"),
        make_item("e3", "C2", 0.3, body=[0.71, 0.71], gt_id="3"),
    ]
    policy = default_policy()
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("overlap", min_sec=0.0, max_sec=1.0),
        "UNK",
        1,
        policy=policy,
        return_debug_bundle=True,
    )
    assert rows[2]["identity_status"] == "pending"
    assert any(log["decision"] == "pending" and log["reason_code"] == "pending_created" for log in debug["decision_logs"])
    assert any(log["decision"] == "pending_gc" and log["reason_code"] == "pending_timeout_gc" for log in debug["decision_logs"])


def test_pending_buffer_can_turn_ambiguous_case_into_reuse():
    items = [
        make_item("e1", "C1", 0.0, body=[1.0, 0.0]),
        make_item("e2", "C3", 0.1, body=[0.0, 1.0], gt_id="2"),
        make_item("e3", "C2", 0.3, body=[0.71, 0.71], gt_id="3"),
    ]
    items[2]["pending_buffer_items"] = [
        {
            "event": {
                **items[2]["event"],
                "pending_buffer_frame_id": 4,
                "pending_buffer_relative_sec": 0.4,
            },
            "face_embedding": None,
            "face_status": "missing",
            "face_count": 0,
            "face_det_score": 0.0,
            "face_bbox": "",
            "face_message": "missing",
            "used_face_crop": "",
            "used_face_crop_path": "",
            "body_embedding": np.asarray([0.98, 0.02], dtype=np.float32),
            "body_status": "ok",
            "body_message": "ok",
            "body_shape": "64x128",
        }
    ]
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("overlap", min_sec=0.0, max_sec=1.0),
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[2]["unknown_global_id"] == rows[0]["unknown_global_id"]
    assert any(log["decision"] == "pending" and log["reason_code"] == "pending_created" for log in debug["decision_logs"])
    assert any(log["decision"] == "unknown_reuse" and log.get("pending_resolution") == "reuse_existing_unknown" for log in debug["decision_logs"])


def test_pending_buffer_can_still_end_with_create_new():
    items = [
        make_item("e1", "C1", 0.0, body=[1.0, 0.0]),
        make_item("e2", "C3", 0.1, body=[0.0, 1.0], gt_id="2"),
        make_item("e3", "C2", 0.3, body=[0.71, 0.71], gt_id="3"),
    ]
    items[2]["pending_buffer_items"] = [
        {
            "event": {
                **items[2]["event"],
                "pending_buffer_frame_id": 4,
                "pending_buffer_relative_sec": 0.4,
            },
            "face_embedding": None,
            "face_status": "missing",
            "face_count": 0,
            "face_det_score": 0.0,
            "face_bbox": "",
            "face_message": "missing",
            "used_face_crop": "",
            "used_face_crop_path": "",
            "body_embedding": np.asarray([0.705, 0.709], dtype=np.float32),
            "body_status": "ok",
            "body_message": "ok",
            "body_shape": "64x128",
        }
    ]
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("overlap", min_sec=0.0, max_sec=1.0),
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[2]["resolution_source"] == "model_unknown_new_profile"
    assert any(log["decision"] == "pending" and log["reason_code"] == "pending_created" for log in debug["decision_logs"])
    assert any(log["decision"] == "create_new" and log.get("pending_resolution") == "create_new_unknown" for log in debug["decision_logs"])


def test_pending_without_future_evidence_is_garbage_collected():
    items = [
        make_item("e1", "C1", 0.0, body=[1.0, 0.0]),
        make_item("e2", "C3", 0.1, body=[0.0, 1.0], gt_id="2"),
        make_item("e3", "C2", 0.3, body=[0.71, 0.71], gt_id="3"),
    ]
    rows, profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("overlap", min_sec=0.0, max_sec=1.0),
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[2]["identity_status"] == "pending"
    assert len(profiles) == 2
    assert debug["pending_runtime"]["stale_pending_count_remaining"] == 0
    assert any(log["decision"] == "pending_gc" for log in debug["decision_logs"])


def test_body_fallback_reuses_when_face_is_low_quality():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0]),
        make_item("e2", "C2", 0.2, face=[1.0, 0.0], body=[1.0, 0.0], face_det_score=0.05),
    ]
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("overlap", min_sec=0.0, max_sec=1.0),
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] == rows[1]["unknown_global_id"]
    assert rows[1]["body_fallback_used"] is True
    assert rows[1]["modality_primary_used"] == "body"
    assert debug["decision_logs"][1]["body_fallback_used"] is True
    assert debug["decision_logs"][1]["face_unusable_reason"] == "face_quality_below_reliable_threshold"


def test_weak_link_requires_strong_appearance_but_can_reuse():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0]),
        make_item("e2", "C2", 1.0, face=[0.98, 0.02], body=[0.98, 0.02]),
    ]
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        make_topology("weak_link", min_sec=0.0, max_sec=2.0),
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] == rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["relation_type"] == "weak_link"


def test_gallery_lifecycle_ttl_and_top_k_refs():
    item1 = make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0], bbox_area=3000)
    policy = {"top_k_face_refs": 2, "top_k_body_refs": 2, "ttl_sec": 1.0, "face_quality_area_bonus": 0.001}
    profile = create_unknown_profile("UNK_0001", item1, policy=policy)
    item2 = make_item("e2", "C1", 0.2, face=[0.9, 0.1], body=[0.9, 0.1], bbox_area=3500)
    item3 = make_item("e3", "C1", 0.4, face=[0.8, 0.2], body=[0.8, 0.2], bbox_area=4000)
    update_unknown_profile(profile, item2, policy=policy)
    update_unknown_profile(profile, item3, policy=policy)
    assert len(profile["face_refs"]) == 2
    assert len(profile["body_refs"]) == 2
    assert profile["representative_face_embedding"] is not None
    active, expired = expire_profiles([profile], current_sec=2.0)
    assert active == []
    assert len(expired) == 1


def test_zone_compatible_transition_reuses_same_unknown_id():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0], zone_id="z_exit", subzone_id="s_exit"),
        make_item("e2", "C2", 0.2, face=[0.99, 0.01], body=[0.99, 0.01], zone_id="z_entry", subzone_id="s_entry"),
    ]
    topology = make_transition_topology(
        "overlap",
        allowed_exit_zones=["z_exit"],
        allowed_entry_zones=["z_entry"],
        allowed_exit_subzones=["s_exit"],
        allowed_entry_subzones=["s_entry"],
        min_sec=0.0,
        max_sec=1.0,
    )
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        topology,
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] == rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["zone_valid"] is True
    assert debug["decision_logs"][1]["transition_rule_used"] == "C1_to_C2_overlap"


def test_zone_mismatch_blocks_reuse_even_with_good_appearance():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0], zone_id="z_exit", subzone_id="s_exit"),
        make_item("e2", "C2", 0.2, face=[0.99, 0.01], body=[0.99, 0.01], zone_id="wrong_zone", subzone_id="s_entry"),
    ]
    topology = make_transition_topology(
        "overlap",
        allowed_exit_zones=["z_exit"],
        allowed_entry_zones=["z_entry"],
        allowed_exit_subzones=["s_exit"],
        allowed_entry_subzones=["s_entry"],
        min_sec=0.0,
        max_sec=1.0,
    )
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        topology,
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] != rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["candidate_evaluations"][0]["zone_valid"] is False
    assert debug["decision_logs"][1]["reason_code"] == "zone_entry_reject"


def test_zone_match_but_out_of_time_window_does_not_reuse():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0], zone_id="z_exit", subzone_id="s_exit"),
        make_item("e2", "C2", 4.0, face=[1.0, 0.0], body=[1.0, 0.0], zone_id="z_entry", subzone_id="s_entry"),
    ]
    topology = make_transition_topology(
        "sequential",
        allowed_exit_zones=["z_exit"],
        allowed_entry_zones=["z_entry"],
        allowed_exit_subzones=["s_exit"],
        allowed_entry_subzones=["s_entry"],
        min_sec=1.0,
        max_sec=2.0,
        avg_sec=1.5,
    )
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        topology,
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] != rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["candidate_evaluations"][0]["time_valid"] is False
    assert debug["decision_logs"][1]["reason_code"] == "sequential_window_reject"


def test_weak_link_zone_compatible_reuses_when_appearance_is_strong():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0], zone_id="z_exit", subzone_id="s_exit"),
        make_item("e2", "C2", 1.0, face=[0.98, 0.02], body=[0.98, 0.02], zone_id="z_entry", subzone_id="s_entry"),
    ]
    topology = make_transition_topology(
        "weak_link",
        allowed_exit_zones=["z_exit"],
        allowed_entry_zones=["z_entry"],
        allowed_exit_subzones=["s_exit"],
        allowed_entry_subzones=["s_entry"],
        min_sec=0.0,
        max_sec=2.0,
        avg_sec=1.0,
    )
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        topology,
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] == rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["relation_type"] == "weak_link"


def test_missing_zone_metadata_falls_back_safely():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0], zone_id=""),
        make_item("e2", "C2", 0.2, face=[0.99, 0.01], body=[0.99, 0.01], zone_id=""),
    ]
    topology = make_transition_topology(
        "overlap",
        allowed_exit_zones=["z_exit"],
        allowed_entry_zones=["z_entry"],
        min_sec=0.0,
        max_sec=1.0,
    )
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        topology,
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] == rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["fallback_without_zone"] is True


def test_subzone_mismatch_blocks_reuse_even_when_zone_matches():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0], zone_id="z_exit", subzone_id="s_exit"),
        make_item("e2", "C2", 0.2, face=[0.99, 0.01], body=[0.99, 0.01], zone_id="z_entry", subzone_id="wrong_subzone"),
    ]
    topology = make_transition_topology(
        "overlap",
        allowed_exit_zones=["z_exit"],
        allowed_entry_zones=["z_entry"],
        allowed_exit_subzones=["s_exit"],
        allowed_entry_subzones=["s_entry"],
        min_sec=0.0,
        max_sec=1.0,
    )
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        topology,
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] != rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["candidate_evaluations"][0]["subzone_valid"] is False
    assert debug["decision_logs"][1]["reason_code"] == "subzone_entry_reject"


def test_missing_subzone_metadata_falls_back_safely():
    items = [
        make_item("e1", "C1", 0.0, face=[1.0, 0.0], body=[1.0, 0.0], zone_id="z_exit", subzone_id=""),
        make_item("e2", "C2", 0.2, face=[0.99, 0.01], body=[0.99, 0.01], zone_id="z_entry", subzone_id=""),
    ]
    topology = make_transition_topology(
        "overlap",
        allowed_exit_zones=["z_exit"],
        allowed_entry_zones=["z_entry"],
        allowed_exit_subzones=["s_exit"],
        allowed_entry_subzones=["s_entry"],
        min_sec=0.0,
        max_sec=1.0,
    )
    rows, _profiles, _trace, debug = assign_model_identities(
        items,
        {},
        topology,
        "UNK",
        1,
        policy=default_policy(),
        return_debug_bundle=True,
    )
    assert rows[0]["unknown_global_id"] == rows[1]["unknown_global_id"]
    assert debug["decision_logs"][1]["fallback_without_subzone"] is True


def test_event_generation_audit_records_subzone_assignment():
    transition_map = {
        "cameras": {
            "C1": {
                "default_zone_id": "z1",
                "default_subzone_id": "s1",
                "zones": [
                    {"zone_id": "z1", "zone_type": "entry", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "priority": 1}
                ],
                "subzones": [
                    {
                        "subzone_id": "s1",
                        "parent_zone_id": "z1",
                        "subzone_type": "entry",
                        "polygon": [[0, 0], [5, 0], [5, 10], [0, 10]],
                        "priority": 100,
                    }
                ],
            }
        }
    }
    spatial = resolve_spatial_context("C1", 2.0, 4.0, transition_map)
    event = {
        "event_id": "evt_1",
        "event_type": "ENTRY_IN",
        "camera_id": "C1",
        "relative_sec": 1.2,
        **spatial,
    }
    row = build_event_assignment_audit_row("mode_b_true_assoc", event)
    assert row["zone_id"] == "z1"
    assert row["subzone_id"] == "s1"
    assert row["subzone_type"] == "entry"
    assert row["matched_subzone_region_id"] == "s1"
