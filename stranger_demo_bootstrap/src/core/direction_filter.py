from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class Point:
    x: float
    y: float


def line_crossing_direction(previous: Point, current: Point, line_y: float) -> str:
    if previous.y >= line_y and current.y < line_y:
        return "IN"
    if previous.y < line_y and current.y >= line_y:
        return "OUT"
    return "UNKNOWN"


def centroid_vector_direction(previous: Point, current: Point, in_vector: Iterable[float]) -> str:
    dx = current.x - previous.x
    dy = current.y - previous.y
    vx, vy = list(in_vector)
    dot = dx * vx + dy * vy
    if dot > 0:
        return "IN"
    if dot < 0:
        return "OUT"
    return "UNKNOWN"


def roi_transition_direction(previous_inside: bool, current_inside: bool) -> str:
    if (not previous_inside) and current_inside:
        return "IN"
    if previous_inside and (not current_inside):
        return "OUT"
    return "UNKNOWN"
