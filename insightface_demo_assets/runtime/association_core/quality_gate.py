import math

import numpy as np

from .config_loader import DEFAULT_ASSOCIATION_POLICY, deep_merge


def _merge_policy(policy):
    return deep_merge(DEFAULT_ASSOCIATION_POLICY["quality_gate"], policy or {})


def estimate_face_pose(landmarks_5):
    if landmarks_5 is None:
        return {
            "landmarks_available": False,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        }
    points = np.asarray(landmarks_5, dtype=np.float32)
    if points.shape != (5, 2):
        return {
            "landmarks_available": False,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        }

    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0
    face_axis = mouth_mid - eye_mid
    inter_eye = float(np.linalg.norm(right_eye - left_eye))
    eye_to_mouth = float(np.linalg.norm(face_axis))

    if inter_eye <= 1e-6 or eye_to_mouth <= 1e-6:
        return {
            "landmarks_available": False,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        }

    # Yaw is approximated from the nose horizontal offset relative to the vertical midline
    # implied by the eye midpoint and mouth midpoint. The more the nose drifts left/right,
    # the larger the horizontal asymmetry, which correlates with head yaw.
    facial_mid_x = float((eye_mid[0] + mouth_mid[0]) / 2.0)
    yaw_ratio = float((nose[0] - facial_mid_x) / inter_eye)
    yaw_deg = float(yaw_ratio * 90.0)

    # Pitch is approximated from the vertical position of the nose between the eye line and
    # mouth line. A centered frontal face keeps the nose around the middle of that span,
    # while looking up/down shifts the ratio and therefore the estimated pitch.
    pitch_ratio = float((nose[1] - eye_mid[1]) / max(eye_to_mouth, 1e-6))
    pitch_deg = float((pitch_ratio - 0.50) * 80.0)

    eye_vector = right_eye - left_eye
    roll_deg = float(math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0]))))

    return {
        "landmarks_available": True,
        "yaw_deg": round(yaw_deg, 4),
        "pitch_deg": round(pitch_deg, 4),
        "roll_deg": round(roll_deg, 4),
    }


def evaluate_buffered_face_gate(face_result, blur_score, policy=None):
    cfg = _merge_policy(policy)
    pose = estimate_face_pose(face_result.get("landmarks"))
    det_score = float(face_result.get("det_score") or 0.0)
    status = face_result.get("status", "missing")
    reason = "accepted"

    if status != "ok":
        reason = "face_missing"
        accepted = False
    elif cfg.get("require_landmarks_for_face_buffer", True) and not pose["landmarks_available"]:
        reason = "missing_landmarks"
        accepted = False
    elif float(blur_score) < float(cfg.get("min_face_blur_score", 45.0)):
        reason = "blur_reject"
        accepted = False
    elif abs(float(pose["yaw_deg"])) > float(cfg.get("max_abs_yaw_deg", 30.0)):
        reason = "yaw_reject"
        accepted = False
    elif abs(float(pose["pitch_deg"])) > float(cfg.get("max_abs_pitch_deg", 20.0)):
        reason = "pitch_reject"
        accepted = False
    elif abs(float(pose["roll_deg"])) > float(cfg.get("max_abs_roll_deg", 20.0)):
        reason = "roll_reject"
        accepted = False
    else:
        accepted = True

    return {
        "accepted_into_buffer": accepted,
        "reject_reason": "" if accepted else reason,
        "landmarks_available": pose["landmarks_available"],
        "yaw_deg": pose["yaw_deg"],
        "pitch_deg": pose["pitch_deg"],
        "roll_deg": pose["roll_deg"],
        "blur_score": round(float(blur_score), 4),
        "det_score": round(det_score, 4),
    }


def evaluate_quality_gate(item, policy=None):
    cfg = _merge_policy(policy)
    event = item["event"]
    bbox_area = float(event.get("bbox_area") or 0.0)
    face_embedding = item.get("face_embedding")
    body_embedding = item.get("body_embedding")
    face_quality = float(item.get("face_det_score") or 0.0)
    body_quality = min(1.0, bbox_area / max(1.0, float(cfg["full_body_area"])))
    pose_pass = bool(item.get("face_pose_pass", True))
    pose_reason = str(item.get("face_gate_reject_reason") or "")

    has_face = face_embedding is not None
    has_body = body_embedding is not None
    reliable_face = has_face and pose_pass and face_quality >= float(cfg["reliable_face_det_score"])
    strong_face = has_face and pose_pass and face_quality >= float(cfg["strong_face_det_score"])
    reliable_body = has_body and bbox_area >= float(cfg["min_body_area"])

    if not has_face and not has_body:
        return {
            "gate_pass": False,
            "decision_type": "defer",
            "reason_code": "poor_quality_no_embeddings",
            "quality_reliability": 0.0,
            "face_quality": round(face_quality, 4),
            "body_quality": round(body_quality, 4),
            "modality_state": "weak_both",
            "primary_modality": "",
            "face_available": False,
            "body_available": False,
            "face_reliable": False,
            "body_reliable": False,
            "face_pose_pass": pose_pass,
            "face_pose_reject_reason": pose_reason,
        }

    if bbox_area < float(cfg["min_bbox_area"]) and not reliable_face:
        return {
            "gate_pass": False,
            "decision_type": "defer",
            "reason_code": "poor_quality_bbox_too_small",
            "quality_reliability": round(max(face_quality, body_quality), 4),
            "face_quality": round(face_quality, 4),
            "body_quality": round(body_quality, 4),
            "modality_state": "weak_both",
            "primary_modality": "",
            "face_available": has_face,
            "body_available": has_body,
            "face_reliable": reliable_face,
            "body_reliable": reliable_body,
            "face_pose_pass": pose_pass,
            "face_pose_reject_reason": pose_reason,
        }

    if strong_face and reliable_body:
        modality_state = "face_and_body"
        primary_modality = "face"
    elif reliable_face:
        modality_state = "face_only" if not has_body else "face_and_body"
        primary_modality = "face"
    elif reliable_body:
        modality_state = "body_only" if not has_face else "face_and_body"
        primary_modality = "body"
    else:
        modality_state = "weak_both"
        primary_modality = "body" if has_body else "face"

    quality_reliability = max(face_quality, body_quality)
    if has_face and has_body:
        quality_reliability = max(quality_reliability, (face_quality + body_quality) / 2.0)

    return {
        "gate_pass": True,
        "decision_type": "compare",
        "reason_code": "quality_gate_pass",
        "quality_reliability": round(quality_reliability, 4),
        "face_quality": round(face_quality, 4),
        "body_quality": round(body_quality, 4),
        "modality_state": modality_state,
        "primary_modality": primary_modality,
        "face_available": has_face,
        "body_available": has_body,
        "face_reliable": reliable_face,
        "body_reliable": reliable_body,
        "face_pose_pass": pose_pass,
        "face_pose_reject_reason": pose_reason,
    }
