import numpy as np


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    v1 = _normalize(v1)
    v2 = _normalize(v2)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _best_similarity(embedding, refs):
    if embedding is None or not refs:
        return 0.0
    return max(cosine_similarity(embedding, ref["embedding"]) for ref in refs if ref.get("embedding") is not None)


def evaluate_appearance_evidence(item, profile, quality_gate_result):
    face_score = _best_similarity(item.get("face_embedding"), profile.get("face_refs", []))
    body_score = _best_similarity(item.get("body_embedding"), profile.get("body_refs", []))
    face_available = item.get("face_embedding") is not None and bool(profile.get("face_refs"))
    body_available = item.get("body_embedding") is not None and bool(profile.get("body_refs"))

    primary_modality = quality_gate_result.get("primary_modality", "")
    modality_state = quality_gate_result.get("modality_state", "weak_both")

    if primary_modality == "face" and face_available:
        appearance_primary = face_score
        appearance_secondary = body_score if body_available else 0.0
    elif body_available:
        primary_modality = "body"
        appearance_primary = body_score
        appearance_secondary = face_score if face_available else 0.0
    elif face_available:
        primary_modality = "face"
        appearance_primary = face_score
        appearance_secondary = 0.0
    else:
        primary_modality = ""
        appearance_primary = 0.0
        appearance_secondary = 0.0
        modality_state = "weak_both"

    if primary_modality == "face" and body_available:
        modality_state = "face_and_body"
    elif primary_modality == "body" and face_available:
        modality_state = "face_and_body"
    elif primary_modality == "face":
        modality_state = "face_only"
    elif primary_modality == "body":
        modality_state = "body_only"

    return {
        "face_score": round(face_score, 4),
        "body_score": round(body_score, 4),
        "appearance_primary": round(float(appearance_primary), 4),
        "appearance_secondary": round(float(appearance_secondary), 4),
        "primary_modality": primary_modality,
        "modality_state": modality_state,
        "face_available": face_available,
        "body_available": body_available,
        "evidence_reason": "appearance_evidence_ready" if primary_modality else "appearance_missing",
    }
