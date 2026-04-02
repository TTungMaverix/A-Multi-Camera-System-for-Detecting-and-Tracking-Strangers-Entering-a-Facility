from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config.loader import ensure_output_dirs, load_yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build known face DB for the stranger demo.")
    parser.add_argument("--app-config", default="config/app.yaml")
    parser.add_argument("--known-root", default="data/known_db")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def iter_known_images(known_root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if not known_root.exists():
        return items
    for person_dir in sorted(path for path in known_root.iterdir() if path.is_dir()):
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((person_dir.name, image_path))
    return items


def load_face_app(app_config: dict[str, Any]):
    insight_cfg = app_config.get("models", {}).get("insightface", {})
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Khong import duoc InsightFace: {exc}") from exc

    app = FaceAnalysis(
        name=insight_cfg.get("model_name", "buffalo_l"),
        root=insight_cfg.get("model_root", str(Path.home() / ".insightface")),
        providers=insight_cfg.get("providers", ["CPUExecutionProvider"]),
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def largest_face(faces: list[Any]) -> Any | None:
    if not faces:
        return None
    return max(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))


def main() -> None:
    args = parse_args()
    app_config = load_yaml(args.app_config)
    ensure_output_dirs(app_config)

    known_root = PROJECT_ROOT / args.known_root
    records = iter_known_images(known_root)
    manifest_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []

    face_app = None
    if not args.manifest_only:
        face_app = load_face_app(app_config)

    for known_id, image_path in records:
        row = {
            "known_id": known_id,
            "image_path": str(image_path),
            "has_embedding": False,
            "face_count": 0,
        }
        if face_app is not None:
            image = cv2.imread(str(image_path))
            if image is not None:
                faces = face_app.get(image)
                row["face_count"] = len(faces)
                face = largest_face(faces)
                if face is not None:
                    row["has_embedding"] = True
                    embedding_rows.append(
                        {
                            "known_id": known_id,
                            "image_path": str(image_path),
                            "bbox_x1": float(face.bbox[0]),
                            "bbox_y1": float(face.bbox[1]),
                            "bbox_x2": float(face.bbox[2]),
                            "bbox_y2": float(face.bbox[3]),
                            "embedding_json": json.dumps(face.embedding.tolist()),
                        }
                    )
        manifest_rows.append(row)

    storage = app_config.get("storage", {})
    manifest_path = PROJECT_ROOT / storage.get("known_manifest_csv", "data/db/known_face_manifest.csv")
    embeddings_path = PROJECT_ROOT / storage.get("known_embeddings_csv", "data/db/known_face_embeddings.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(embedding_rows).to_csv(embeddings_path, index=False)

    print(
        json.dumps(
            {
                "known_identity_count": len({row["known_id"] for row in manifest_rows}),
                "known_image_count": len(manifest_rows),
                "embedding_count": len(embedding_rows),
                "manifest_path": str(manifest_path),
                "embeddings_path": str(embeddings_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
