from src.core.topology import can_transition, candidate_next_cameras


TOPOLOGY_CONFIG = {
    "graph": {
        "edges": [
            {
                "from": "cam05",
                "to": "cam06",
                "min_travel_sec": 0,
                "avg_travel_sec": 1,
                "max_travel_sec": 4,
                "allow_overlap": True,
                "confidence": 0.97,
            },
            {
                "from": "cam05",
                "to": "cam03",
                "min_travel_sec": 1,
                "avg_travel_sec": 3,
                "max_travel_sec": 10,
                "allow_overlap": False,
                "confidence": 0.88,
            },
        ]
    }
}


def test_candidate_next_cameras_sorted_by_confidence() -> None:
    results = candidate_next_cameras(TOPOLOGY_CONFIG, "cam05")
    assert [item["camera_id"] for item in results] == ["cam06", "cam03"]


def test_can_transition_allows_overlap_at_zero_seconds() -> None:
    assert can_transition(TOPOLOGY_CONFIG, "cam05", "cam06", 0)


def test_can_transition_rejects_too_fast_non_overlap_edge() -> None:
    assert not can_transition(TOPOLOGY_CONFIG, "cam05", "cam03", 0)
