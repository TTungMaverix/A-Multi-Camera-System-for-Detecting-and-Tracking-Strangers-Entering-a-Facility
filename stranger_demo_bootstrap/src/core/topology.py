from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TopologyEdge:
    from_camera: str
    to_camera: str
    min_travel_sec: float
    avg_travel_sec: float
    max_travel_sec: float
    allow_overlap: bool
    confidence: float


def load_edges(topology_config: dict[str, Any]) -> list[TopologyEdge]:
    edges: list[TopologyEdge] = []
    for item in topology_config.get("graph", {}).get("edges", []):
        edges.append(
            TopologyEdge(
                from_camera=item["from"],
                to_camera=item["to"],
                min_travel_sec=float(item["min_travel_sec"]),
                avg_travel_sec=float(item["avg_travel_sec"]),
                max_travel_sec=float(item["max_travel_sec"]),
                allow_overlap=bool(item.get("allow_overlap", False)),
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return edges


def candidate_next_cameras(topology_config: dict[str, Any], camera_id: str) -> list[dict[str, Any]]:
    results = []
    for edge in load_edges(topology_config):
        if edge.from_camera != camera_id:
            continue
        results.append(
            {
                "camera_id": edge.to_camera,
                "min_sec": edge.min_travel_sec,
                "avg_sec": edge.avg_travel_sec,
                "max_sec": edge.max_travel_sec,
                "allow_overlap": edge.allow_overlap,
                "confidence": edge.confidence,
            }
        )
    return sorted(results, key=lambda item: (-item["confidence"], item["avg_sec"]))


def can_transition(
    topology_config: dict[str, Any],
    from_camera: str,
    to_camera: str,
    delta_seconds: float,
) -> bool:
    for edge in load_edges(topology_config):
        if edge.from_camera != from_camera or edge.to_camera != to_camera:
            continue
        if delta_seconds < edge.min_travel_sec:
            return False
        if delta_seconds == 0 and edge.allow_overlap:
            return True
        return delta_seconds <= edge.max_travel_sec
    return False
