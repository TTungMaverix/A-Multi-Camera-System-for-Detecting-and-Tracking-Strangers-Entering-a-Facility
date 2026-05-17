import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from insightface.app import FaceAnalysis

from run_face_resolution_demo import build_gallery_embeddings, resolve_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _safe_name(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "person")).strip("_")
    return value or "person"


def _read_manifest(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["identity_id", "display_name", "source_repo_path", "gallery_rel_path", "seed_type", "status", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _discover_rows(project_root: Path, known_root: Path):
    rows = []
    image_paths = sorted(path for path in known_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    id_by_dir = {}
    for image_path in image_paths:
        try:
            rel_path = image_path.relative_to(project_root)
        except ValueError:
            rel_path = image_path
        parent = image_path.parent
        if parent not in id_by_dir:
            display = _safe_name(parent.name if parent != known_root else image_path.stem)
            id_by_dir[parent] = f"FACILITY_{len(id_by_dir) + 1:03d}_{display}"
        known_id = id_by_dir[parent]
        display_name = known_id.split("_", 2)[-1].replace("_", " ")
        rows.append(
            {
                "identity_id": known_id,
                "display_name": display_name,
                "source_repo_path": str(rel_path),
                "gallery_rel_path": str(rel_path),
                "seed_type": "facility_known_db",
                "status": "pending_embedding",
                "notes": "auto_discovered_from_known_faces_facility",
            }
        )
    return rows


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build the active facility Known Face DB with InsightFace-compatible embeddings.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--known-root", default=os.environ.get("KNOWN_DB_ROOT", r"D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"))
    parser.add_argument("--manifest-csv", default="insightface_demo_assets/known_face_facility_manifest.csv")
    parser.add_argument("--embeddings-csv", default="insightface_demo_assets/runtime/known_face_facility_embeddings.csv")
    parser.add_argument("--summary-json", default="outputs/evaluations/known_facility_db/known_db_build_summary.json")
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    known_root = resolve_path(project_root, args.known_root)
    manifest_csv = resolve_path(project_root, args.manifest_csv)
    embeddings_csv = resolve_path(project_root, args.embeddings_csv)
    summary_json = resolve_path(project_root, args.summary_json)
    known_root.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(manifest_csv)
    usable_rows = [row for row in rows if row.get("gallery_rel_path")]
    if not usable_rows:
        rows = _discover_rows(project_root, known_root)
        _write_manifest(manifest_csv, rows)

    app = FaceAnalysis(
        name=args.model_name,
        root=str(Path(args.model_root).resolve()) if args.model_root else str(Path.home() / ".insightface"),
        providers=[args.provider],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    identity_means, per_image_rows = build_gallery_embeddings(app, rows, project_root, embeddings_csv)
    embedding_dim = 0
    if identity_means:
        first_identity = next(iter(identity_means.values()))
        try:
            embedding_dim = int(len(first_identity))
        except TypeError:
            embedding_dim = 0
    summary = {
        "known_root": str(known_root),
        "manifest_csv": str(manifest_csv),
        "embeddings_csv": str(embeddings_csv),
        "model_name": args.model_name,
        "embedding_dimension": embedding_dim,
        "person_count": len({row.get("identity_id") for row in rows if row.get("identity_id")}),
        "image_count": len(rows),
        "embedding_ok_count": sum(1 for row in per_image_rows if row.get("embedding_status") == "ok"),
        "grayscale_aligned_ok_count": sum(1 for row in per_image_rows if row.get("grayscale_preprocessing_status") == "ok"),
        "reject_reason_counts": {},
        "identity_ids_with_embedding": sorted(identity_means.keys()),
        "rows": per_image_rows,
    }
    for row in per_image_rows:
        if row.get("embedding_status") != "ok":
            reason = row.get("notes") or row.get("embedding_status") or "unknown"
            summary["reject_reason_counts"][reason] = summary["reject_reason_counts"].get(reason, 0) + 1
    save_json(summary_json, summary)
    print(summary_json)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
