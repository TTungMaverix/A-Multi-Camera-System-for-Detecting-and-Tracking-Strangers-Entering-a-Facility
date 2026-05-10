import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_FACE_SIZE = (112, 112)


def load_image_unicode(image_path: Path):
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(image_path: Path, image):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {image_path}")
    encoded.tofile(str(image_path))


def parse_bbox(value):
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    return [int(round(float(item))) for item in value[:4]]


def aligned_grayscale_face(image_path: Path, bbox=None, output_size=DEFAULT_FACE_SIZE):
    """Return an aligned grayscale face tensor from an already face-detected image.

    The current pipeline stores InsightFace bbox/landmark metadata but not a full
    5-point similarity transform. This utility deliberately avoids comparing raw
    unaligned frames: it crops the detected face bbox, resizes to a fixed shape,
    converts to grayscale, and normalizes intensity before pixel-level audit
    scoring.
    """
    image = load_image_unicode(Path(image_path))
    if image is None:
        return None, "failed_to_read_image"
    h, w = image.shape[:2]
    parsed = parse_bbox(bbox)
    if parsed is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = parsed
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "empty_face_crop"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)
    normalized = cv2.equalizeHist(resized)
    return normalized.astype(np.float32) / 255.0, "ok"


def save_aligned_grayscale_face(image_path: Path, output_path: Path, bbox=None, output_size=DEFAULT_FACE_SIZE):
    gray, status = aligned_grayscale_face(image_path, bbox=bbox, output_size=output_size)
    if gray is None:
        return {"status": status, "path": "", "shape": ""}
    write_image_unicode(output_path, (gray * 255.0).clip(0, 255).astype(np.uint8))
    return {"status": "ok", "path": str(output_path), "shape": f"{output_size[0]}x{output_size[1]}"}


def normalized_pixel_similarity(face_a, face_b):
    if face_a is None or face_b is None:
        return 0.0
    a = np.asarray(face_a, dtype=np.float32).reshape(-1)
    b = np.asarray(face_b, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return 0.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    # Map normalized correlation from [-1, 1] into [0, 1] for score fusion/audit.
    return round(float((np.dot(a, b) / denom + 1.0) / 2.0), 4)
