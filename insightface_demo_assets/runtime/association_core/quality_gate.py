from .config_loader import DEFAULT_ASSOCIATION_POLICY, deep_merge


def _merge_policy(policy):
    return deep_merge(DEFAULT_ASSOCIATION_POLICY["quality_gate"], policy or {})


def evaluate_quality_gate(item, policy=None):
    cfg = _merge_policy(policy)
    event = item["event"]
    bbox_area = float(event.get("bbox_area") or 0.0)
    face_embedding = item.get("face_embedding")
    body_embedding = item.get("body_embedding")
    face_quality = float(item.get("face_det_score") or 0.0)
    body_quality = min(1.0, bbox_area / max(1.0, float(cfg["full_body_area"])))

    has_face = face_embedding is not None
    has_body = body_embedding is not None
    reliable_face = has_face and face_quality >= float(cfg["reliable_face_det_score"])
    strong_face = has_face and face_quality >= float(cfg["strong_face_det_score"])
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
    }
