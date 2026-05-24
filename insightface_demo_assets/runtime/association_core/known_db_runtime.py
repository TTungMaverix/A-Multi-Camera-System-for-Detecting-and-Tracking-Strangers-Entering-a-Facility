import csv
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_KNOWN_DB_ROOT = r"D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
DEFAULT_KNOWN_FACE_MANIFEST = "insightface_demo_assets/known_face_facility_manifest.csv"
DEFAULT_KNOWN_FACE_EMBEDDINGS_PKL = "insightface_demo_assets/runtime/known_face_facility_embeddings.pkl"
DEFAULT_KNOWN_FACE_EMBEDDINGS_CSV = "insightface_demo_assets/runtime/known_face_facility_embeddings.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
KNOWN_ID_PATTERN = re.compile(r"(known_\d+)", re.IGNORECASE)
RUNTIME_MANIFEST_FIELDS = [
    "identity_id",
    "display_name",
    "source_repo_path",
    "gallery_rel_path",
    "seed_type",
    "status",
    "notes",
]
DISCOVERY_MANIFEST_FIELDS = [
    "person_id",
    "name",
    "image_path",
    "view_label",
    "source_folder",
]
EMBEDDING_CSV_FIELDS = [
    "identity_id",
    "display_name",
    "image_path",
    "view_label",
    "source_folder",
    "embedding_status",
    "embedding_dim",
    "model_name",
    "embedding_json",
    "det_score",
    "face_bbox",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "notes",
]


def default_known_db_root():
    return os.environ.get("KNOWN_DB_ROOT", DEFAULT_KNOWN_DB_ROOT)


def apply_known_db_defaults(runtime_config):
    updated = dict(runtime_config or {})
    gallery_root = str(updated.get("known_face_gallery_root", "") or "").strip()
    manifest_csv = str(updated.get("known_face_manifest_csv", "") or "").strip()
    embeddings_csv = str(updated.get("known_face_embeddings_csv", "") or "").strip()
    embeddings_pkl = str(updated.get("known_face_embeddings_pkl", "") or "").strip()

    if not gallery_root or gallery_root.replace("\\", "/").endswith("insightface_demo_assets/known_faces"):
        updated["known_face_gallery_root"] = default_known_db_root()
    if not manifest_csv or manifest_csv.replace("\\", "/").endswith("insightface_demo_assets/known_face_manifest.csv"):
        updated["known_face_manifest_csv"] = DEFAULT_KNOWN_FACE_MANIFEST
    if not embeddings_csv or embeddings_csv.replace("\\", "/").endswith("known_face_embeddings_template.csv"):
        updated["known_face_embeddings_csv"] = DEFAULT_KNOWN_FACE_EMBEDDINGS_CSV
    if not embeddings_pkl:
        updated["known_face_embeddings_pkl"] = DEFAULT_KNOWN_FACE_EMBEDDINGS_PKL
    return updated


def _safe_display_name(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "person")).strip("_")
    return value or "person"


def _read_manifest_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_manifest_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _canonical_known_id_from_path(path: Path):
    # Prefer the nearest parent folder so a mislabeled filename inside a
    # canonical directory (for example known_055/known_054_front_face.jpg)
    # still stays grouped under known_055.
    for parent in [path.parent, *path.parents]:
        match = KNOWN_ID_PATTERN.search(str(parent.name))
        if match:
            return match.group(1).lower()
    for value in [path.stem, path.name]:
        match = KNOWN_ID_PATTERN.search(str(value))
        if match:
            return match.group(1).lower()
    return ""


def _normalize_view_label(value: str, canonical_id: str):
    raw = str(value or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    known_prefix_match = KNOWN_ID_PATTERN.match(lowered)
    if known_prefix_match:
        lowered = lowered[len(known_prefix_match.group(1)) :]
    elif canonical_id and lowered.startswith(canonical_id):
        lowered = lowered[len(canonical_id) :]
    lowered = re.sub(r"^[\/_\-\s]+", "", lowered)
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return lowered


def discover_known_face_rows(known_root: Path):
    if not known_root.exists():
        raise FileNotFoundError(f"Known root does not exist: {known_root}")
    if not known_root.is_dir():
        raise NotADirectoryError(f"Known root is not a directory: {known_root}")

    rows = []
    non_standard_count = 0
    image_paths = sorted(
        path.resolve()
        for path in known_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No image files found under Known root: {known_root}")

    for image_path in image_paths:
        canonical_id = _canonical_known_id_from_path(image_path)
        non_standard = False
        if not canonical_id:
            non_standard = True
            canonical_id = image_path.parent.name.strip() or image_path.stem.strip() or "unknown_person"
            non_standard_count += 1
        relative_parent = image_path.parent.relative_to(known_root)
        stem_has_canonical_id = bool(canonical_id and canonical_id in image_path.stem.lower())
        view_label = _normalize_view_label(image_path.stem, canonical_id)
        if relative_parent != Path(".") and (not view_label or not stem_has_canonical_id):
            parent_view_label = _normalize_view_label(relative_parent.name, canonical_id)
            if parent_view_label:
                view_label = parent_view_label
        rows.append(
            {
                "person_id": canonical_id,
                "name": canonical_id,
                "image_path": str(image_path),
                "view_label": view_label,
                "source_folder": str(relative_parent).replace("\\", "/"),
                "non_standard_known_id": non_standard,
            }
        )
    return rows, {
        "row_count": len(rows),
        "person_count": len({row["person_id"] for row in rows}),
        "sample_first_10_person_ids": sorted({row["person_id"] for row in rows})[:10],
        "non_standard_known_id_count": non_standard_count,
    }


def _runtime_row_from_discovery_row(project_root: Path, known_root: Path, row):
    image_path = Path(str(row.get("image_path", "") or ""))
    if not image_path:
        return None
    if not image_path.is_absolute():
        image_path = (known_root / image_path).resolve()
    if not image_path.exists():
        return None
    try:
        rel_path = image_path.relative_to(project_root)
        rel_value = str(rel_path).replace("\\", "/")
    except ValueError:
        rel_value = str(image_path)
    identity_id = str(row.get("person_id", "") or "").strip()
    display_name = str(row.get("name", "") or identity_id).strip() or identity_id
    view_label = str(row.get("view_label", "") or "").strip()
    source_folder = str(row.get("source_folder", "") or "").strip()
    notes = []
    if view_label:
        notes.append(f"view_label={view_label}")
    if source_folder:
        notes.append(f"source_folder={source_folder}")
    if row.get("non_standard_known_id"):
        notes.append("non_standard_known_id")
    return {
        "identity_id": identity_id,
        "display_name": display_name,
        "source_repo_path": str(image_path),
        "gallery_rel_path": rel_value,
        "seed_type": "facility_known_db",
        "status": "pending_embedding",
        "notes": ";".join(notes) if notes else "auto_discovered_from_known_db_root",
    }


def _runtime_row_from_manifest_row(project_root: Path, known_root: Path, row):
    if row.get("gallery_rel_path") and row.get("identity_id"):
        return {
            "identity_id": str(row.get("identity_id", "") or "").strip(),
            "display_name": str(row.get("display_name", "") or row.get("identity_id", "")).strip(),
            "source_repo_path": str(row.get("source_repo_path", "") or row.get("gallery_rel_path", "")).strip(),
            "gallery_rel_path": str(row.get("gallery_rel_path", "") or "").strip(),
            "seed_type": str(row.get("seed_type", "") or "facility_known_db").strip(),
            "status": str(row.get("status", "") or "pending_embedding").strip(),
            "notes": str(row.get("notes", "") or "").strip(),
        }
    if row.get("person_id") and row.get("image_path"):
        return _runtime_row_from_discovery_row(project_root, known_root, row)
    return None


def ensure_known_db_manifest_rows(project_root: Path, manifest_csv: Path, known_root: Path):
    raw_rows = _read_manifest_rows(manifest_csv)
    rows = []
    for raw_row in raw_rows:
        runtime_row = _runtime_row_from_manifest_row(project_root, known_root, raw_row)
        if runtime_row and runtime_row.get("gallery_rel_path"):
            rows.append(runtime_row)

    auto_discovered = False
    discovery_info = {
        "row_count": 0,
        "person_count": 0,
        "sample_first_10_person_ids": [],
        "non_standard_known_id_count": 0,
    }
    if not rows and known_root.exists():
        discovered_rows, discovery_info = discover_known_face_rows(known_root)
        rows = [
            runtime_row
            for runtime_row in (
                _runtime_row_from_discovery_row(project_root, known_root, row) for row in discovered_rows
            )
            if runtime_row is not None
        ]
        if rows:
            _write_manifest_rows(manifest_csv, rows, RUNTIME_MANIFEST_FIELDS)
            auto_discovered = True

    return rows, {
        "auto_discovered": auto_discovered,
        "manifest_exists": manifest_csv.exists(),
        "identity_count": len({row.get("identity_id", "") for row in rows if row.get("identity_id")}),
        "image_count": len(rows),
        "sample_first_10_person_ids": discovery_info.get("sample_first_10_person_ids", []),
        "non_standard_known_id_count": int(discovery_info.get("non_standard_known_id_count", 0)),
    }


def parse_manifest_for_build(project_root: Path, manifest_csv: Path, known_root: Path):
    raw_rows = _read_manifest_rows(manifest_csv)
    discovered_rows = []
    if raw_rows:
        for raw_row in raw_rows:
            if raw_row.get("person_id") and raw_row.get("image_path"):
                image_path = Path(str(raw_row.get("image_path", "") or ""))
                if not image_path.is_absolute():
                    image_path = (project_root / image_path).resolve()
                if image_path.exists():
                    discovered_rows.append(
                        {
                            "person_id": str(raw_row.get("person_id", "") or "").strip(),
                            "name": str(raw_row.get("name", "") or raw_row.get("person_id", "")).strip(),
                            "image_path": str(image_path),
                            "view_label": str(raw_row.get("view_label", "") or "").strip(),
                            "source_folder": str(raw_row.get("source_folder", "") or "").strip(),
                            "non_standard_known_id": "non_standard_known_id" in str(raw_row.get("notes", "") or ""),
                        }
                    )
            else:
                runtime_row = _runtime_row_from_manifest_row(project_root, known_root, raw_row)
                if runtime_row is None:
                    continue
                image_path = Path(runtime_row["gallery_rel_path"])
                if not image_path.is_absolute():
                    image_path = (project_root / image_path).resolve()
                if not image_path.exists():
                    continue
                view_label = ""
                source_folder = ""
                notes = str(runtime_row.get("notes", "") or "")
                for note in notes.split(";"):
                    if note.startswith("view_label="):
                        view_label = note.split("=", 1)[1]
                    elif note.startswith("source_folder="):
                        source_folder = note.split("=", 1)[1]
                discovered_rows.append(
                    {
                        "person_id": runtime_row["identity_id"],
                        "name": runtime_row["display_name"] or runtime_row["identity_id"],
                        "image_path": str(image_path),
                        "view_label": view_label,
                        "source_folder": source_folder,
                        "non_standard_known_id": "non_standard_known_id" in notes,
                    }
                )
    else:
        discovered_rows, _summary = discover_known_face_rows(known_root)

    if not discovered_rows:
        raise RuntimeError(
            f"No valid manifest rows or discoverable known images were found. manifest={manifest_csv} known_root={known_root}"
        )
    return discovered_rows


def write_discovery_manifest(path: Path, rows):
    cleaned_rows = []
    for row in rows:
        cleaned_rows.append(
            {
                "person_id": row.get("person_id", ""),
                "name": row.get("name", ""),
                "image_path": row.get("image_path", ""),
                "view_label": row.get("view_label", ""),
                "source_folder": row.get("source_folder", ""),
            }
        )
    _write_manifest_rows(path, cleaned_rows, DISCOVERY_MANIFEST_FIELDS)


def write_embedding_csv(path: Path, rows):
    _write_manifest_rows(path, rows, EMBEDDING_CSV_FIELDS)


def write_embedding_pkl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _normalize_embedding_record(record):
    identity_id = str(record.get("identity_id", "") or record.get("person_id", "")).strip()
    if not identity_id:
        return None
    raw_embedding = record.get("embedding")
    if raw_embedding is None:
        raw_embedding = record.get("embedding_json", "")
    if isinstance(raw_embedding, str):
        if not raw_embedding.strip():
            return None
        try:
            raw_embedding = json.loads(raw_embedding)
        except json.JSONDecodeError:
            return None
    vector = np.asarray(raw_embedding, dtype=np.float32)
    if vector.size == 0:
        return None
    display_name = str(record.get("display_name", "") or record.get("name", "") or identity_id).strip() or identity_id
    image_path = str(record.get("image_path", "") or record.get("image_rel_path", "") or "").strip()
    return {
        "identity_id": identity_id,
        "display_name": display_name,
        "image_path": image_path,
        "view_label": str(record.get("view_label", "") or "").strip(),
        "source_folder": str(record.get("source_folder", "") or "").strip(),
        "embedding": vector,
        "embedding_dim": int(record.get("embedding_dimension", record.get("embedding_dim", vector.size)) or vector.size),
        "model_name": str(record.get("model_name", "") or "buffalo_l").strip() or "buffalo_l",
        "det_score": float(record.get("det_score", 0.0) or 0.0),
        "face_bbox": str(record.get("face_bbox", record.get("bbox", "")) or "").strip(),
    }


def build_gallery_index(records):
    gallery = {}
    for record in records:
        normalized = _normalize_embedding_record(record)
        if normalized is None:
            continue
        identity_id = normalized["identity_id"]
        entry = gallery.setdefault(
            identity_id,
            {
                "display_name": normalized["display_name"],
                "embeddings": [],
                "refs": [],
            },
        )
        entry["embeddings"].append(normalized["embedding"])
        entry["refs"].append(
            {
                "image_path": normalized["image_path"],
                "view_label": normalized["view_label"],
                "source_folder": normalized["source_folder"],
                "det_score": normalized["det_score"],
                "face_bbox": normalized["face_bbox"],
                "embedding_dim": normalized["embedding_dim"],
                "model_name": normalized["model_name"],
            }
        )
    for entry in gallery.values():
        if entry["embeddings"]:
            entry["embedding"] = np.mean(np.stack(entry["embeddings"], axis=0), axis=0).astype(np.float32)
            entry["embedding_dim"] = int(entry["embeddings"][0].shape[0])
            entry["embedding_count"] = len(entry["embeddings"])
            entry["model_name"] = entry["refs"][0].get("model_name", "buffalo_l") if entry["refs"] else "buffalo_l"
    return gallery


def load_known_face_embeddings(base_dir: Path, embeddings_pkl: Path | None = None, embeddings_csv: Path | None = None):
    records = []
    source_path = ""
    source_format = ""

    if embeddings_pkl and embeddings_pkl.exists():
        with embeddings_pkl.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            records = list(payload.get("records", []) or [])
        elif isinstance(payload, list):
            records = payload
        source_path = str(embeddings_pkl)
        source_format = "pkl"
    elif embeddings_csv and embeddings_csv.exists():
        with embeddings_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("embedding_status", "") or "").strip() != "ok":
                    continue
                records.append(row)
        source_path = str(embeddings_csv)
        source_format = "csv"

    gallery = build_gallery_index(records)
    embedding_dims = sorted(
        {
            int(entry.get("embedding_dim", 0) or 0)
            for entry in gallery.values()
            if int(entry.get("embedding_dim", 0) or 0) > 0
        }
    )
    identity_ids = sorted(gallery.keys())
    embedding_count = sum(len(entry.get("embeddings", [])) for entry in gallery.values())
    summary = {
        "source_path": source_path,
        "source_format": source_format,
        "identity_count": len(identity_ids),
        "embedding_count": embedding_count,
        "embedding_dimension": embedding_dims[0] if len(embedding_dims) == 1 else (embedding_dims or []),
        "sample_first_10_person_ids": identity_ids[:10],
    }
    return gallery, summary
