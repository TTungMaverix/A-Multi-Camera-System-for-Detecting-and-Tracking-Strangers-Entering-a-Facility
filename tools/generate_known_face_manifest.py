import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "insightface_demo_assets" / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from association_core.known_db_runtime import discover_known_face_rows, write_discovery_manifest  # noqa: E402


DEFAULT_KNOWN_ROOT = Path(r"D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID")
DEFAULT_OUT_CSV = Path("insightface_demo_assets/known_face_facility_manifest.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate known face manifest CSV from Known ID folders.")
    parser.add_argument("--known-root", default=str(DEFAULT_KNOWN_ROOT))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    return parser.parse_args()


def safe_text(value):
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def main():
    args = parse_args()
    known_root = Path(args.known_root).resolve()
    out_csv = (REPO_ROOT / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv).resolve()
    rows, summary = discover_known_face_rows(known_root)
    write_discovery_manifest(out_csv, rows)
    print(f"KNOWN_ROOT={safe_text(known_root)}")
    print(f"OUT_CSV={safe_text(out_csv)}")
    print(f"ROW_COUNT={summary['row_count']}")
    print(f"PERSON_COUNT={summary['person_count']}")
    print(f"SAMPLE_FIRST_10_PERSON_IDS={','.join(summary['sample_first_10_person_ids'])}")
    print(f"NON_STANDARD_KNOWN_ID_COUNT={summary['non_standard_known_id_count']}")


if __name__ == "__main__":
    main()
