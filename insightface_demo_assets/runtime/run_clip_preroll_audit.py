import argparse
from pathlib import Path

from run_face_resolution_demo import read_csv, save_json, write_csv


def _to_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _boolish(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _camera_summary(run_output_root: Path, camera_id: str):
    track_csv = run_output_root / "tracks" / f"{camera_id}_tracks.csv"
    entry_csv = run_output_root / "events" / "entry_in_events.csv"
    track_rows = read_csv(track_csv)
    entry_rows = [row for row in read_csv(entry_csv) if row.get("camera_id") == camera_id]
    if not track_rows:
        return {
            "camera_id": camera_id,
            "track_row_count": 0,
            "entry_in_count": len(entry_rows),
            "status": "no_tracks",
            "recommendation": "Verify detector output or source-video mapping before calibrating further.",
        }
    first_row = sorted(
        track_rows,
        key=lambda row: (_to_int(row.get("source_frame_id_actual"), 999999), _to_int(row.get("frame_id"), 999999)),
    )[0]
    first_source_frame = _to_int(first_row.get("source_frame_id_actual"), -1)
    first_inside = _boolish(first_row.get("roi_footpoint_inside"))
    first_entry = min([_to_float(row.get("relative_sec"), 0.0) for row in entry_rows], default=None)
    status = "event_ok" if entry_rows else "event_missing"
    recommendation = "No action required."
    if first_source_frame <= 0 and first_inside and not entry_rows:
        status = "late_start"
        recommendation = "Clip starts with the person already inside ROI. Add 3-5 seconds of pre-roll or rely on the late-start fallback."
    elif first_source_frame <= 6 and first_inside and not entry_rows:
        status = "starts_inside_roi"
        recommendation = "Clip starts too close to the entry boundary. Capture an earlier segment before the person enters ROI."
    elif not entry_rows:
        recommendation = "Review manual ROI/entry-line calibration and direction history on the first observed track."
    return {
        "camera_id": camera_id,
        "track_row_count": len(track_rows),
        "unique_track_count": len({row.get("local_track_id", "") for row in track_rows if row.get("local_track_id")}),
        "entry_in_count": len(entry_rows),
        "first_source_frame_id_actual": first_source_frame,
        "first_frame_id": _to_int(first_row.get("frame_id"), -1),
        "first_relative_sec": _to_float(first_row.get("relative_sec"), 0.0),
        "first_local_track_id": first_row.get("local_track_id", ""),
        "first_detection_score": _to_float(first_row.get("detection_score"), 0.0),
        "first_roi_footpoint_inside": first_inside,
        "first_entry_in_relative_sec": first_entry,
        "status": status,
        "recommendation": recommendation,
    }


def _resolve_output_root_from_clip(project_root: Path, clip: str):
    normalized = str(clip or "").strip().lower()
    if normalized == "b2":
        return project_root / "outputs" / "evaluations" / "b2_known_face_runtime_validation" / "offline_runs" / "b2"
    if normalized == "c1":
        return project_root / "outputs" / "evaluations" / "c1_demo_artifacts" / "offline_runs" / "c1"
    return project_root / "outputs" / "offline_runs" / normalized


def main():
    parser = argparse.ArgumentParser(description="Audit whether a clip likely starts too late for clean ENTRY_IN generation.")
    parser.add_argument("--run-output-root", default="")
    parser.add_argument("--clip", default="")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--summary-csv", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    run_output_root = Path(args.run_output_root).resolve() if args.run_output_root else None
    if run_output_root is None:
        if not args.clip:
            raise SystemExit("Provide either --run-output-root or --clip.")
        run_output_root = _resolve_output_root_from_clip(project_root, args.clip).resolve()

    summaries = []
    track_dir = run_output_root / "tracks"
    if track_dir.exists():
        for track_csv in sorted(track_dir.glob("*_tracks.csv")):
            camera_id = track_csv.stem.replace("_tracks", "")
            summaries.append(_camera_summary(run_output_root, camera_id))
    if not summaries:
        for camera_id in ("C1", "C2", "C3", "C4"):
            summaries.append(
                {
                    "camera_id": camera_id,
                    "track_row_count": 0,
                    "entry_in_count": 0,
                    "status": "no_tracks",
                    "recommendation": "No track file found. Verify output root and offline pipeline run.",
                }
            )

    summary_json = Path(args.summary_json).resolve() if args.summary_json else run_output_root / "summaries" / "preroll_audit.json"
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else run_output_root / "summaries" / "preroll_audit.csv"
    save_json(summary_json, summaries)
    fieldnames = []
    for row in summaries:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(summary_csv, summaries, fieldnames)
    print(summary_json)


if __name__ == "__main__":
    main()
