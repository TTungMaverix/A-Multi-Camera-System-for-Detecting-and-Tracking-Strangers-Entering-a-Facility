import cv2
import numpy as np

from association_core.decision_policy import _decision_score_components
from association_core.face_pixel import aligned_grayscale_face, normalized_pixel_similarity


def test_aligned_grayscale_face_resizes_and_equalizes(tmp_path):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = (80, 140, 220)
    image_path = tmp_path / "face.png"
    assert cv2.imwrite(str(image_path), image)

    gray, status = aligned_grayscale_face(image_path, bbox=(30, 20, 70, 60), output_size=(32, 32))

    assert status == "ok"
    assert gray.shape == (32, 32)
    assert gray.dtype == np.float32
    assert gray.mean() > 0


def test_normalized_pixel_similarity_prefers_identical_faces():
    face_a = np.tile(np.linspace(0, 255, 32, dtype=np.float32), (32, 1))
    face_b = face_a.copy()
    face_c = np.zeros((32, 32), dtype=np.uint8)
    face_c[:, 16:] = 255

    assert normalized_pixel_similarity(face_a, face_b) == 1.0
    assert normalized_pixel_similarity(face_a, face_c) < 1.0


def test_decision_score_components_report_weighted_formula():
    policy = {
        "decision_score_weights": {
            "face": 0.5,
            "body": 0.2,
            "time": 0.2,
            "topology": 0.1,
        },
        "decision_score_threshold": 0.7,
    }
    candidate = {
        "face_score": 0.8,
        "body_score": 0.6,
        "time_score": 1.0,
        "topology_score": 1.0,
    }

    score = _decision_score_components(candidate, policy)

    assert score["FaceScore"] == 0.8
    assert score["BodyScore"] == 0.6
    assert score["TimeScore"] == 1.0
    assert score["TopologyScore"] == 1.0
    assert score["final_score"] == 0.82
    assert score["score_threshold"] == 0.7
    assert "FaceScore" in score["score_formula"]
