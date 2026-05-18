from pathlib import Path

from association_core.known_db_runtime import (
    DEFAULT_KNOWN_DB_ROOT,
    DEFAULT_KNOWN_FACE_EMBEDDINGS,
    DEFAULT_KNOWN_FACE_MANIFEST,
    apply_known_db_defaults,
    ensure_known_db_manifest_rows,
)


def test_apply_known_db_defaults_populates_expected_paths():
    payload = apply_known_db_defaults({})
    assert payload["known_face_gallery_root"] == DEFAULT_KNOWN_DB_ROOT
    assert payload["known_face_manifest_csv"] == DEFAULT_KNOWN_FACE_MANIFEST
    assert payload["known_face_embeddings_csv"] == DEFAULT_KNOWN_FACE_EMBEDDINGS


def test_ensure_known_db_manifest_rows_discovers_identity_folders(tmp_path: Path):
    known_root = tmp_path / "Known ID"
    person_dir = known_root / "known_001"
    person_dir.mkdir(parents=True)
    image_path = person_dir / "face_01.jpg"
    image_path.write_bytes(b"fake")
    manifest_csv = tmp_path / "known_face_facility_manifest.csv"

    rows, info = ensure_known_db_manifest_rows(tmp_path, manifest_csv, known_root)

    assert info["auto_discovered"] is True
    assert info["identity_count"] == 1
    assert info["image_count"] == 1
    assert rows[0]["identity_id"].startswith("FACILITY_001_known_001")
    normalized_rel = rows[0]["gallery_rel_path"].replace("\\", "/")
    assert normalized_rel.endswith("Known ID/known_001/face_01.jpg")
    assert manifest_csv.exists()
