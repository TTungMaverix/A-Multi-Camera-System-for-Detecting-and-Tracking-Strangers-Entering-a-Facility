import csv
import json
from pathlib import Path

from association_core.decision_policy import best_known_match
from association_core.known_db_runtime import (
    discover_known_face_rows,
    ensure_known_db_manifest_rows,
    load_known_face_embeddings,
    write_embedding_csv,
)


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def test_discover_known_face_rows_canonicalizes_known_ids(tmp_path):
    known_root = tmp_path / "Known ID"
    _touch(known_root / "known_001_front_face" / "img1.jpg")
    _touch(known_root / "known_001" / "known_001_side_1.png")
    _touch(known_root / "nested" / "known_002_left" / "img2.jpeg")
    _touch(known_root / "person_misc" / "misc.png")

    rows, summary = discover_known_face_rows(known_root)

    assert len(rows) == 4
    assert summary["person_count"] == 3
    assert summary["non_standard_known_id_count"] == 1
    assert sorted({row["person_id"] for row in rows}) == ["known_001", "known_002", "person_misc"]
    assert any(row["view_label"] == "front_face" for row in rows if row["person_id"] == "known_001")
    assert any(row["view_label"] == "side_1" for row in rows if row["person_id"] == "known_001")


def test_ensure_known_db_manifest_rows_accepts_simple_manifest(tmp_path):
    project_root = tmp_path
    known_root = tmp_path / "Known ID"
    image_path = known_root / "known_001_front_face.jpg"
    _touch(image_path)
    manifest_csv = tmp_path / "known_face_facility_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["person_id", "name", "image_path"])
        writer.writeheader()
        writer.writerow(
            {
                "person_id": "known_001",
                "name": "known_001",
                "image_path": str(image_path.resolve()),
            }
        )

    rows, info = ensure_known_db_manifest_rows(project_root, manifest_csv, known_root)

    assert info["image_count"] == 1
    assert rows[0]["identity_id"] == "known_001"
    resolved_gallery_path = Path(rows[0]["gallery_rel_path"])
    if not resolved_gallery_path.is_absolute():
        resolved_gallery_path = (project_root / resolved_gallery_path).resolve()
    assert resolved_gallery_path == image_path.resolve()


def test_load_known_face_embeddings_supports_best_of_gallery(tmp_path):
    embeddings_csv = tmp_path / "known_face_facility_embeddings.csv"
    write_embedding_csv(
        embeddings_csv,
        [
            {
                "identity_id": "known_001",
                "display_name": "known_001",
                "image_path": "img_front.jpg",
                "view_label": "front_face",
                "source_folder": "known_001_front_face",
                "embedding_status": "ok",
                "embedding_dim": 2,
                "model_name": "buffalo_l",
                "embedding_json": json.dumps([1.0, 0.0]),
                "det_score": 0.99,
                "face_bbox": "[0,0,10,10]",
                "bbox_width": 10,
                "bbox_height": 10,
                "bbox_area": 100,
                "notes": "ok",
            },
            {
                "identity_id": "known_001",
                "display_name": "known_001",
                "image_path": "img_side.jpg",
                "view_label": "side_1",
                "source_folder": "known_001_side",
                "embedding_status": "ok",
                "embedding_dim": 2,
                "model_name": "buffalo_l",
                "embedding_json": json.dumps([0.0, 1.0]),
                "det_score": 0.99,
                "face_bbox": "[0,0,10,10]",
                "bbox_width": 10,
                "bbox_height": 10,
                "bbox_area": 100,
                "notes": "ok",
            },
            {
                "identity_id": "known_002",
                "display_name": "known_002",
                "image_path": "img_other.jpg",
                "view_label": "front_face",
                "source_folder": "known_002",
                "embedding_status": "ok",
                "embedding_dim": 2,
                "model_name": "buffalo_l",
                "embedding_json": json.dumps([0.5, 0.5]),
                "det_score": 0.99,
                "face_bbox": "[0,0,10,10]",
                "bbox_width": 10,
                "bbox_height": 10,
                "bbox_area": 100,
                "notes": "ok",
            },
        ],
    )

    gallery, summary = load_known_face_embeddings(tmp_path, embeddings_csv=embeddings_csv)
    match = best_known_match([0.0, 1.0], gallery)

    assert summary["identity_count"] == 2
    assert summary["embedding_count"] == 3
    assert match["identity_id"] == "known_001"
    assert match["best_image_path"] == "img_side.jpg"
