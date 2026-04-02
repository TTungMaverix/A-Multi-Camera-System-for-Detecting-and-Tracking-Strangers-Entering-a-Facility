from src.core.association import choose_unknown_match, weighted_score


APP_CONFIG = {
    "thresholds": {"unknown_association": 0.52},
    "association": {
        "weights": {
            "face_score": 0.55,
            "body_score": 0.15,
            "time_score": 0.15,
            "topology_score": 0.15,
        },
        "fallback_no_face": {
            "face_score": 0.0,
            "body_score": 0.45,
            "time_score": 0.30,
            "topology_score": 0.25,
        },
    },
}

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
            }
        ]
    }
}


def test_weighted_score_uses_primary_weights_when_face_exists() -> None:
    score = weighted_score(APP_CONFIG, 0.9, 0.2, 0.6, 1.0, face_available=True)
    assert round(score, 4) == 0.765


def test_choose_unknown_match_returns_best_candidate_above_threshold() -> None:
    event = {"camera_id": "cam06", "time_delta_seconds": 0}
    candidates = [
        {
            "global_id": "unknown_01",
            "last_camera": "cam05",
            "face_score": 0.9,
            "body_score": 0.2,
            "time_score": 0.6,
            "face_available": True,
        },
        {
            "global_id": "unknown_02",
            "last_camera": "cam05",
            "face_score": 0.6,
            "body_score": 0.2,
            "time_score": 0.4,
            "face_available": True,
        },
    ]
    match = choose_unknown_match(APP_CONFIG, TOPOLOGY_CONFIG, event, candidates)
    assert match is not None
    assert match["global_id"] == "unknown_01"
