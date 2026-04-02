from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .topology import can_transition


@dataclass(slots=True)
class AssociationWeights:
    face_score: float
    body_score: float
    time_score: float
    topology_score: float


def load_weights(app_config: dict[str, Any], fallback: bool = False) -> AssociationWeights:
    block_name = "fallback_no_face" if fallback else "weights"
    values = app_config.get("association", {}).get(block_name, {})
    return AssociationWeights(
        face_score=float(values.get("face_score", 0.0)),
        body_score=float(values.get("body_score", 0.0)),
        time_score=float(values.get("time_score", 0.0)),
        topology_score=float(values.get("topology_score", 0.0)),
    )


def weighted_score(
    app_config: dict[str, Any],
    face_score: float,
    body_score: float,
    time_score: float,
    topology_score: float,
    face_available: bool = True,
) -> float:
    weights = load_weights(app_config, fallback=not face_available)
    return (
        face_score * weights.face_score
        + body_score * weights.body_score
        + time_score * weights.time_score
        + topology_score * weights.topology_score
    )


def topology_score_from_edge(
    topology_config: dict[str, Any],
    from_camera: str,
    to_camera: str,
    delta_seconds: float,
) -> float:
    return 1.0 if can_transition(topology_config, from_camera, to_camera, delta_seconds) else 0.0


def choose_unknown_match(
    app_config: dict[str, Any],
    topology_config: dict[str, Any],
    event: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    threshold = float(app_config.get("thresholds", {}).get("unknown_association", 0.52))
    best: dict[str, Any] | None = None

    for candidate in candidates:
        face_available = bool(candidate.get("face_available", True))
        topo = topology_score_from_edge(
            topology_config=topology_config,
            from_camera=str(candidate["last_camera"]),
            to_camera=str(event["camera_id"]),
            delta_seconds=float(event["time_delta_seconds"]),
        )
        score = weighted_score(
            app_config=app_config,
            face_score=float(candidate.get("face_score", 0.0)),
            body_score=float(candidate.get("body_score", 0.0)),
            time_score=float(candidate.get("time_score", 0.0)),
            topology_score=topo,
            face_available=face_available,
        )
        enriched = dict(candidate)
        enriched["association_score"] = round(score, 6)
        enriched["topology_score"] = topo
        if score < threshold:
            continue
        if best is None or enriched["association_score"] > best["association_score"]:
            best = enriched
    return best
