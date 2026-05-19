import argparse
import csv
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_KNOWN_ROOT = Path(r"D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID")
DEFAULT_OUT_CSV = Path("insightface_demo_assets/known_face_facility_manifest.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate known face manifest CSV from Known ID folders.")
    parser.add_argument("--known-root", default=str(DEFAULT_KNOWN_ROOT))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    return parser.parse_args()


def discover_rows(known_root: Path):
    if not known_root.exists():
        raise FileNotFoundError(f"Known root does not exist: {known_root}")
    if not known_root.is_dir():
        raise NotADirectoryError(f"Known root is not a directory: {known_root}")

    rows = []
    for person_dir in sorted(path for path in known_root.iterdir() if path.is_dir()):
        person_id = person_dir.name
        image_paths = sorted(
            path.resolve()
            for path in person_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            rows.append(
                {
                    "person_id": person_id,
                    "name": person_id,
                    "image_path": str(image_path),
                }
            )
    if not rows:
        raise RuntimeError(f"No image files found under Known root: {known_root}")
    return rows


def write_manifest(out_csv: Path, rows):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["person_id", "name", "image_path"])
        writer.writeheader()
        writer.writerows(rows)


def safe_text(value):
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def main():
    args = parse_args()
    known_root = Path(args.known_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    rows = discover_rows(known_root)
    write_manifest(out_csv, rows)
    unique_people = sorted({row["person_id"] for row in rows})
    print(f"KNOWN_ROOT={safe_text(known_root)}")
    print(f"OUT_CSV={safe_text(out_csv)}")
    print(f"ROW_COUNT={len(rows)}")
    print(f"PERSON_COUNT={len(unique_people)}")


if __name__ == "__main__":
    main()
