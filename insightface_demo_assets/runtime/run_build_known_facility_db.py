import argparse
import csv
import json
import os
import sys
from pathlib import Path

from insightface.app import FaceAnalysis

from association_core.known_db_runtime import (
    DEFAULT_KNOWN_DB_ROOT,
    ensure_known_db_manifest_rows,
)
from run_face_resolution_demo import build_gallery_embeddings, resolve_path


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build the active facility Known Face DB with InsightFace-compatible embeddings.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--known-root", default=os.environ.get("KNOWN_DB_ROOT", DEFAULT_KNOWN_DB_ROOT))
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

    rows, manifest_info = ensure_known_db_manifest_rows(project_root, manifest_csv, known_root)

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
        "manifest_auto_discovered": bool(manifest_info.get("auto_discovered", False)),
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
