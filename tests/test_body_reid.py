from pathlib import Path

import cv2
import numpy as np

from association_core.body_reid import build_tracklet_body_representation


def _write_image(path: Path, image):
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def _pattern_image(width=72, height=160, brightness=160):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (brightness, brightness - 20, brightness - 40)
    for y in range(0, height, 16):
        image[y : y + 8, :, :] = (brightness - 60, brightness, brightness - 10)
    for x in range(0, width, 14):
        image[:, x : x + 4, :] = (brightness, brightness - 70, brightness)
    return image


class FakeBodyExtractor:
    def __init__(self):
        self.cfg = {
            "extractor_name": "fake_body",
            "min_crop_width": 20,
            "min_crop_height": 40,
            "tracklet_pooling_max_candidates": 10,
            "tracklet_pooling_top_k": 3,
            "tracklet_pooling_min_blur_score": 25.0,
            "tracklet_pooling_min_bbox_area": 1800.0,
            "tracklet_pooling_min_relative_bbox_area": 0.55,
            "clahe_enabled": False,
            "gray_world_normalization": False,
        }

    def extract_array(self, image, source_path=""):
        mean_channels = image.mean(axis=(0, 1)).astype(np.float32)
        norm = np.linalg.norm(mean_channels)
        embedding = mean_channels if norm <= 1e-6 else (mean_channels / norm)
        return {
            "status": "ok",
            "embedding": embedding.astype(np.float32),
            "message": "ok_fake_body",
            "shape": f"{image.shape[1]}x{image.shape[0]}",
        }


def test_tracklet_body_representation_pools_multiple_valid_crops(tmp_path):
    extractor = FakeBodyExtractor()
    rows = []
    for index, brightness in enumerate((140, 150, 160), start=1):
        crop_path = tmp_path / f"body_{index}.png"
        _write_image(crop_path, _pattern_image(brightness=brightness))
        rows.append(
            {
                "body_crop_path": str(crop_path),
                "bbox_area": 6000 + (index * 500),
                "frame_id": index,
                "relative_sec": float(index) / 10.0,
            }
        )

    result = build_tracklet_body_representation(rows, body_reid_runtime=extractor)

    assert result["status"] == "ok"
    assert result["embedding"] is not None
    assert result["tracklet_valid_crop_count"] == 3
    assert result["tracklet_selected_crop_count"] == 3
    assert len(result["tracklet_selected_crop_paths"]) == 3
    assert result["tracklet_reject_counts"] == {}


def test_tracklet_body_representation_rejects_small_and_unstable_crops(tmp_path):
    extractor = FakeBodyExtractor()
    valid_path = tmp_path / "valid.png"
    small_path = tmp_path / "small.png"
    unstable_path = tmp_path / "unstable.png"

    _write_image(valid_path, _pattern_image(width=72, height=160, brightness=150))
    _write_image(small_path, _pattern_image(width=16, height=32, brightness=150))
    _write_image(unstable_path, _pattern_image(width=72, height=160, brightness=150))

    rows = [
        {"body_crop_path": str(valid_path), "bbox_area": 7000, "frame_id": 1, "relative_sec": 0.1},
        {"body_crop_path": str(small_path), "bbox_area": 500, "frame_id": 2, "relative_sec": 0.2},
        {"body_crop_path": str(unstable_path), "bbox_area": 3000, "frame_id": 3, "relative_sec": 0.3},
    ]

    result = build_tracklet_body_representation(rows, body_reid_runtime=extractor)

    assert result["status"] == "ok"
    assert result["tracklet_valid_crop_count"] == 1
    assert result["tracklet_selected_crop_count"] == 1
    assert result["tracklet_reject_counts"]["too_small"] >= 1
    assert result["tracklet_reject_counts"]["bbox_unstable"] >= 1
