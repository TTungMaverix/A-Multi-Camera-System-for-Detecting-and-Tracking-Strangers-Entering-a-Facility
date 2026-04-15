from association_core.quality_gate import estimate_face_pose, evaluate_buffered_face_gate


def test_pose_gate_accepts_frontal_face_with_good_blur():
    face_result = {
        "status": "ok",
        "det_score": 0.92,
        "landmarks": [
            [30.0, 40.0],
            [70.0, 40.0],
            [50.0, 58.0],
            [36.0, 82.0],
            [64.0, 82.0],
        ],
    }
    gate = evaluate_buffered_face_gate(face_result, blur_score=180.0)
    assert gate["accepted_into_buffer"] is True
    assert abs(gate["yaw_deg"]) <= 30.0
    assert abs(gate["pitch_deg"]) <= 20.0


def test_pose_gate_rejects_large_yaw_even_when_image_is_sharp():
    face_result = {
        "status": "ok",
        "det_score": 0.95,
        "landmarks": [
            [30.0, 40.0],
            [70.0, 40.0],
            [72.0, 58.0],
            [36.0, 82.0],
            [64.0, 82.0],
        ],
    }
    pose = estimate_face_pose(face_result["landmarks"])
    gate = evaluate_buffered_face_gate(face_result, blur_score=220.0)
    assert abs(pose["yaw_deg"]) > 30.0
    assert gate["accepted_into_buffer"] is False
    assert gate["reject_reason"] == "yaw_reject"


def test_pose_gate_rejects_large_pitch_even_when_image_is_sharp():
    face_result = {
        "status": "ok",
        "det_score": 0.95,
        "landmarks": [
            [30.0, 40.0],
            [70.0, 40.0],
            [50.0, 80.0],
            [36.0, 82.0],
            [64.0, 82.0],
        ],
    }
    pose = estimate_face_pose(face_result["landmarks"])
    gate = evaluate_buffered_face_gate(face_result, blur_score=220.0)
    assert abs(pose["pitch_deg"]) > 20.0
    assert gate["accepted_into_buffer"] is False
    assert gate["reject_reason"] == "pitch_reject"
