import numpy as np

from .config_loader import DEFAULT_ASSOCIATION_POLICY, deep_merge


def _merge_policy(policy):
    return deep_merge(DEFAULT_ASSOCIATION_POLICY["appearance_evidence"], policy or {})


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


def _combined_similarity(embedding, refs, representative, reference_weight, representative_weight):
    ref_score = _best_similarity(embedding, refs)
    rep_score = cosine_similarity(embedding, representative) if representative is not None else 0.0
    if representative is None:
        combined = ref_score
    elif not refs:
        combined = rep_score
    else:
        combined = (float(reference_weight) * ref_score) + (float(representative_weight) * rep_score)
    return round(float(combined), 4), round(float(ref_score), 4), round(float(rep_score), 4)


def evaluate_appearance_evidence(item, profile, quality_gate_result):
    cfg = _merge_policy(quality_gate_result.get("appearance_evidence_policy"))
    face_score, face_ref_score, face_rep_score = _combined_similarity(
        item.get("face_embedding"),
        profile.get("face_refs", []),
        profile.get("representative_face_embedding"),
        cfg["face_reference_weight"],
        cfg["face_representative_weight"],
    )
    body_score, body_ref_score, body_rep_score = _combined_similarity(
        item.get("body_embedding"),
        profile.get("body_refs", []),
        profile.get("representative_body_embedding"),
        cfg["body_reference_weight"],
        cfg["body_representative_weight"],
    )
    face_available = item.get("face_embedding") is not None and (
        bool(profile.get("face_refs")) or profile.get("representative_face_embedding") is not None
    )
    body_available = item.get("body_embedding") is not None and (
        bool(profile.get("body_refs")) or profile.get("representative_body_embedding") is not None
    )
    face_reliable = bool(quality_gate_result.get("face_reliable")) and face_available
    body_reliable = bool(quality_gate_result.get("body_reliable")) and body_available

    primary_modality = quality_gate_result.get("primary_modality", "")
    modality_state = quality_gate_result.get("modality_state", "weak_both")
    secondary_modality = ""
    secondary_reliable = False
    face_unusable_reason = ""
    body_fallback_used = False

    if cfg["prefer_primary_face_when_reliable"] and primary_modality == "face" and face_reliable:
        appearance_primary = face_score
        appearance_secondary = body_score if body_available else 0.0
        secondary_modality = "body" if body_available else ""
        secondary_reliable = body_reliable
    elif cfg["allow_body_primary_fallback"] and body_available:
        primary_modality = "body"
        appearance_primary = body_score
        appearance_secondary = face_score if face_available else 0.0
        secondary_modality = "face" if face_available else ""
        secondary_reliable = face_reliable
        body_fallback_used = not face_reliable
    elif face_available:
        primary_modality = "face"
        appearance_primary = face_score
        appearance_secondary = body_score if body_available else 0.0
        secondary_modality = "body" if body_available else ""
        secondary_reliable = body_reliable
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

    if body_fallback_used:
        if item.get("face_embedding") is None:
            face_unusable_reason = "face_embedding_missing"
        elif not bool(quality_gate_result.get("face_reliable")):
            face_unusable_reason = "face_quality_below_reliable_threshold"
        elif not face_available:
            face_unusable_reason = "no_face_gallery_reference"
        else:
            face_unusable_reason = "body_fallback_selected"
    elif primary_modality == "face":
        face_unusable_reason = ""
    elif item.get("face_embedding") is None:
        face_unusable_reason = "face_embedding_missing"
    elif not bool(quality_gate_result.get("face_reliable")):
        face_unusable_reason = "face_quality_below_reliable_threshold"

    return {
        "face_score": round(face_score, 4),
        "body_score": round(body_score, 4),
        "face_ref_score": face_ref_score,
        "face_representative_score": face_rep_score,
        "body_ref_score": body_ref_score,
        "body_representative_score": body_rep_score,
        "appearance_primary": round(float(appearance_primary), 4),
        "appearance_secondary": round(float(appearance_secondary), 4),
        "primary_modality": primary_modality,
        "secondary_modality": secondary_modality,
        "modality_state": modality_state,
        "face_available": face_available,
        "body_available": body_available,
        "face_reliable": face_reliable,
        "body_reliable": body_reliable,
        "secondary_reliable": secondary_reliable,
        "body_fallback_used": body_fallback_used,
        "face_unusable_reason": face_unusable_reason,
        "appearance_evidence_policy": cfg,
        "evidence_reason": (
            "body_fallback_evidence_ready"
            if body_fallback_used
            else ("appearance_evidence_ready" if primary_modality else "appearance_missing")
        ),
    }
