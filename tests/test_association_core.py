import numpy as np

from association_core.config_loader import load_association_policy
from association_core.decision_policy import assign_model_identities
from association_core.gallery_lifecycle import create_unknown_profile, expire_profiles, update_unknown_profile
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


def make_item(event_id, camera_id, sec, face=None, body=None, bbox_area=4000, gt_id="1", zone_id=""):
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
        },
        "face_embedding": np.asarray(face, dtype=np.float32) if face is not None else None,
        "face_status": "ok" if face is not None else "missing",
        "face_count": 1 if face is not None else 0,
        "face_det_score": 0.9 if face is not None else 0.0,
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


def test_ambiguous_margin_creates_new_unknown_by_policy():
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
    assert rows[2]["resolution_source"] == "model_unknown_new_profile"
    assert debug["decision_logs"][2]["decision"] == "create_new"
    assert "ambiguous" in debug["decision_logs"][2]["reason_code"]


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
