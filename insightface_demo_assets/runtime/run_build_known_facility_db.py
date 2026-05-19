import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from insightface.app import FaceAnalysis

from association_core.known_db_runtime import (
    DEFAULT_KNOWN_DB_ROOT,
    DEFAULT_KNOWN_FACE_EMBEDDINGS_CSV,
    DEFAULT_KNOWN_FACE_EMBEDDINGS_PKL,
    DEFAULT_KNOWN_FACE_MANIFEST,
    discover_known_face_rows,
    parse_manifest_for_build,
    write_discovery_manifest,
    write_embedding_csv,
    write_embedding_pkl,
)
from run_face_resolution_demo import extract_embedding_from_image


CONFIG_DEFAULT = Path(__file__).with_name("face_demo_config.json")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_path(project_root: Path, value: str):
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Build known face embeddings from Known ID root or manifest.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--known-root", default=DEFAULT_KNOWN_DB_ROOT)
    parser.add_argument("--manifest-csv", default=DEFAULT_KNOWN_FACE_MANIFEST)
    parser.add_argument("--embeddings-pkl", default=DEFAULT_KNOWN_FACE_EMBEDDINGS_PKL)
    parser.add_argument("--embeddings-csv", default=DEFAULT_KNOWN_FACE_EMBEDDINGS_CSV)
    parser.add_argument("--summary-json", default="outputs/evaluations/known_facility_db/known_db_build_summary.json")
    return parser.parse_args()


def build_face_app(config_path: Path):
    config = load_json(config_path)
    runtime_cfg = config.get("insightface_runtime", {}) or {}
    app = FaceAnalysis(
        name=runtime_cfg.get("recommended_model_name", "buffalo_l"),
        root=str(Path(runtime_cfg.get("recommended_model_root", str(Path.home() / ".insightface")))),
        providers=[runtime_cfg.get("provider", "CPUExecutionProvider")],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app, runtime_cfg.get("recommended_model_name", "buffalo_l")


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    known_root = resolve_path(project_root, args.known_root)
    manifest_csv = resolve_path(project_root, args.manifest_csv)
    embeddings_pkl = resolve_path(project_root, args.embeddings_pkl)
    embeddings_csv = resolve_path(project_root, args.embeddings_csv)
    summary_json = resolve_path(project_root, args.summary_json)

    if not known_root.exists():
        raise SystemExit(f"Known root does not exist: {known_root}")

    if not manifest_csv.exists() or not manifest_csv.read_text(encoding="utf-8-sig").strip():
        discovered_rows, _summary = discover_known_face_rows(known_root)
        write_discovery_manifest(manifest_csv, discovered_rows)

    manifest_rows = parse_manifest_for_build(project_root, manifest_csv, known_root)
    if not manifest_rows:
        raise SystemExit("No manifest rows available for Known DB build.")

    app, model_name = build_face_app(CONFIG_DEFAULT)

    image_count_per_person = Counter(row["person_id"] for row in manifest_rows)
    discovered_person_ids = sorted(image_count_per_person.keys())
    known_id_dirs = sorted(
        {
            path.name.lower()
            for path in known_root.rglob("*")
            if path.is_dir() and path.name.lower().startswith("known_")
        }
    )
    empty_known_id_dirs = sorted(set(known_id_dirs) - set(discovered_person_ids))

    csv_rows = []
    pkl_records = []
    reject_reason_counts = Counter()
    valid_embedding_counts = Counter()
    embedding_dimensions = Counter()

    for row in manifest_rows:
        image_path = Path(row["image_path"]).resolve()
        person_id = row["person_id"]
        face_result = extract_embedding_from_image(app, image_path)
        status = str(face_result.get("status", "missing") or "missing")
        note = str(face_result.get("message", "") or status)
        csv_row = {
            "identity_id": person_id,
            "display_name": row.get("name", person_id),
            "image_path": str(image_path),
            "view_label": row.get("view_label", ""),
            "source_folder": row.get("source_folder", ""),
            "embedding_status": status,
            "embedding_dim": len(face_result["embedding"]) if face_result.get("embedding") is not None else "",
            "model_name": model_name,
            "embedding_json": json.dumps(face_result["embedding"].tolist()) if face_result.get("embedding") is not None else "",
            "det_score": round(float(face_result.get("det_score", 0.0) or 0.0), 6),
            "face_bbox": face_result.get("bbox", ""),
            "bbox_width": int(face_result.get("bbox_width", 0) or 0),
            "bbox_height": int(face_result.get("bbox_height", 0) or 0),
            "bbox_area": int(face_result.get("bbox_area", 0) or 0),
            "notes": note,
        }
        csv_rows.append(csv_row)
        if face_result.get("embedding") is None:
            reject_reason_counts[status or "embedding_missing"] += 1
            continue
        embedding = face_result["embedding"].tolist()
        pkl_records.append(
            {
                "identity_id": person_id,
                "display_name": row.get("name", person_id),
                "image_path": str(image_path),
                "view_label": row.get("view_label", ""),
                "source_folder": row.get("source_folder", ""),
                "embedding_dimension": len(embedding),
                "model_name": model_name,
                "det_score": round(float(face_result.get("det_score", 0.0) or 0.0), 6),
                "face_bbox": face_result.get("bbox", ""),
                "embedding": embedding,
            }
        )
        valid_embedding_counts[person_id] += 1
        embedding_dimensions[len(embedding)] += 1

    if not pkl_records:
        reject_summary = dict(sorted(reject_reason_counts.items()))
        save_json(
            summary_json,
            {
                "status": "failed",
                "known_root": str(known_root),
                "manifest_csv": str(manifest_csv),
                "embeddings_pkl": str(embeddings_pkl),
                "embeddings_csv": str(embeddings_csv),
                "model_name": model_name,
                "total_person_count": len(discovered_person_ids),
                "total_image_count": len(manifest_rows),
                "image_ok_count": 0,
                "image_failed_count": len(manifest_rows),
                "embedding_ok_count": 0,
                "embedding_failed_count": len(manifest_rows),
                "image_count_per_person": dict(sorted(image_count_per_person.items())),
                "min_images_per_person": min(image_count_per_person.values()) if image_count_per_person else 0,
                "max_images_per_person": max(image_count_per_person.values()) if image_count_per_person else 0,
                "avg_images_per_person": round(sum(image_count_per_person.values()) / max(len(image_count_per_person), 1), 3),
                "persons_with_zero_valid_embeddings": discovered_person_ids,
                "reject_reason_counts": reject_summary,
                "sample_first_10_person_ids": discovered_person_ids[:10],
                "non_standard_known_id_count": sum(1 for row in manifest_rows if row.get("non_standard_known_id")),
                "known_id_dir_count_detected": len(known_id_dirs),
                "empty_known_id_dir_count": len(empty_known_id_dirs),
                "warning": "FAILED_NO_VALID_EMBEDDINGS",
                "multi_face_selection_rule": "largest_area_times_det_score",
            },
        )
        raise SystemExit("Known DB build failed: no valid embeddings were produced.")

    write_embedding_csv(embeddings_csv, csv_rows)
    write_embedding_pkl(
        embeddings_pkl,
        {
            "model_name": model_name,
            "embedding_dimension": max(embedding_dimensions, key=embedding_dimensions.get),
            "records": pkl_records,
        },
    )

    zero_embedding_ids = sorted(person_id for person_id in discovered_person_ids if valid_embedding_counts.get(person_id, 0) == 0)
    warning = ""
    if len(discovered_person_ids) < 50 and len(known_id_dirs) >= 50:
        warning = "DISCOVERED_PERSON_COUNT_LOW_VS_KNOWN_ID_DIR_COUNT"

    summary = {
        "status": "ok",
        "known_root": str(known_root),
        "manifest_csv": str(manifest_csv),
        "embeddings_pkl": str(embeddings_pkl),
        "embeddings_csv": str(embeddings_csv),
        "model_name": model_name,
        "embedding_dimension": max(embedding_dimensions, key=embedding_dimensions.get),
        "total_person_count": len(discovered_person_ids),
        "total_image_count": len(manifest_rows),
        "image_ok_count": len(pkl_records),
        "image_failed_count": len(manifest_rows) - len(pkl_records),
        "embedding_ok_count": len(pkl_records),
        "embedding_failed_count": len(manifest_rows) - len(pkl_records),
        "image_count_per_person": dict(sorted(image_count_per_person.items())),
        "min_images_per_person": min(image_count_per_person.values()) if image_count_per_person else 0,
        "max_images_per_person": max(image_count_per_person.values()) if image_count_per_person else 0,
        "avg_images_per_person": round(sum(image_count_per_person.values()) / max(len(image_count_per_person), 1), 3),
        "persons_with_zero_valid_embeddings": zero_embedding_ids,
        "reject_reason_counts": dict(sorted(reject_reason_counts.items())),
        "sample_first_10_person_ids": discovered_person_ids[:10],
        "non_standard_known_id_count": sum(1 for row in manifest_rows if row.get("non_standard_known_id")),
        "known_id_dir_count_detected": len(known_id_dirs),
        "empty_known_id_dir_count": len(empty_known_id_dirs),
        "warning": warning,
        "multi_face_selection_rule": "largest_area_times_det_score",
    }
    save_json(summary_json, summary)

    print(f"KNOWN_ROOT={known_root}")
    print(f"MANIFEST_CSV={manifest_csv}")
    print(f"EMBEDDINGS_PKL={embeddings_pkl}")
    print(f"EMBEDDINGS_CSV={embeddings_csv}")
    print(f"TOTAL_PERSON_COUNT={summary['total_person_count']}")
    print(f"TOTAL_IMAGE_COUNT={summary['total_image_count']}")
    print(f"EMBEDDING_OK_COUNT={summary['embedding_ok_count']}")
    print(f"EMBEDDING_FAILED_COUNT={summary['embedding_failed_count']}")
    print(f"EMBEDDING_DIMENSION={summary['embedding_dimension']}")


if __name__ == "__main__":
    main()
