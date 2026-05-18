import csv
import os
import re
from pathlib import Path


DEFAULT_KNOWN_DB_ROOT = r"D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
DEFAULT_KNOWN_FACE_MANIFEST = "insightface_demo_assets/known_face_facility_manifest.csv"
DEFAULT_KNOWN_FACE_EMBEDDINGS = "insightface_demo_assets/runtime/known_face_facility_embeddings.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST_FIELDS = [
    "identity_id",
    "display_name",
    "source_repo_path",
    "gallery_rel_path",
    "seed_type",
    "status",
    "notes",
]


def default_known_db_root():
    return os.environ.get("KNOWN_DB_ROOT", DEFAULT_KNOWN_DB_ROOT)


def apply_known_db_defaults(runtime_config):
    updated = dict(runtime_config or {})
    updated.setdefault("known_face_gallery_root", default_known_db_root())
    updated.setdefault("known_face_manifest_csv", DEFAULT_KNOWN_FACE_MANIFEST)
    updated.setdefault("known_face_embeddings_csv", DEFAULT_KNOWN_FACE_EMBEDDINGS)
    return updated


def safe_identity_name(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "person")).strip("_")
    return value or "person"


def read_manifest_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MANIFEST_FIELDS})


def discover_known_db_manifest_rows(project_root: Path, known_root: Path):
    rows = []
    image_paths = sorted(
        path for path in known_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    id_by_dir = {}
    for image_path in image_paths:
        try:
            rel_path = image_path.relative_to(project_root)
        except ValueError:
            rel_path = image_path
        parent = image_path.parent
        if parent not in id_by_dir:
            display = safe_identity_name(parent.name if parent != known_root else image_path.stem)
            id_by_dir[parent] = f"FACILITY_{len(id_by_dir) + 1:03d}_{display}"
        known_id = id_by_dir[parent]
        rows.append(
            {
                "identity_id": known_id,
                "display_name": known_id.split("_", 2)[-1].replace("_", " "),
                "source_repo_path": str(rel_path),
                "gallery_rel_path": str(rel_path),
                "seed_type": "facility_known_db",
                "status": "pending_embedding",
                "notes": "auto_discovered_from_known_db_root",
            }
        )
    return rows


def ensure_known_db_manifest_rows(project_root: Path, manifest_csv: Path, known_root: Path):
    rows = [row for row in read_manifest_rows(manifest_csv) if row.get("gallery_rel_path")]
    auto_discovered = False
    if not rows and known_root.exists():
        rows = discover_known_db_manifest_rows(project_root, known_root)
        if rows:
            write_manifest_rows(manifest_csv, rows)
            auto_discovered = True
    info = {
        "auto_discovered": auto_discovered,
        "manifest_exists": manifest_csv.exists(),
        "identity_count": len({row.get("identity_id", "") for row in rows if row.get("identity_id")}),
        "image_count": len(rows),
    }
    return rows, info
