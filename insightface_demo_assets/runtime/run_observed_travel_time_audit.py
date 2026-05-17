import argparse
import csv
import json
from pathlib import Path

import yaml


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_transition_map(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = payload.get("camera_transition_map", payload)
    transitions = {}
    for row in root.get("transitions", []) or []:
        transitions[(row.get("src_camera_id", ""), row.get("dst_camera_id", ""))] = row
    return transitions


def load_resolved_rows(run_output_root: Path):
    candidate_paths = [
        run_output_root / "events" / "resolved_events.csv",
        run_output_root / "runtime" / "resolved_events_mode_b_true_assoc.csv",
        run_output_root / "resolved_events_mode_b_true_assoc.csv",
        run_output_root / "runtime" / "resolved_events.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return load_rows(path), path
    return [], candidate_paths[0]


def observed_deltas(rows, transitions):
    grouped = {}
    for row in rows:
        identity_id = row.get("unknown_global_id") or row.get("resolved_global_id") or ""
        camera_id = row.get("camera_id", "")
        if not identity_id or not camera_id or row.get("identity_status") != "unknown":
            continue
        grouped.setdefault(identity_id, []).append(row)
    observations = []
    for identity_id, items in grouped.items():
        ordered = sorted(items, key=lambda row: float(row.get("relative_sec", 0.0) or 0.0))
        for left, right in zip(ordered, ordered[1:]):
            key = (left.get("camera_id", ""), right.get("camera_id", ""))
            transition = transitions.get(key, {})
            delta = round(float(right.get("relative_sec", 0.0) or 0.0) - float(left.get("relative_sec", 0.0) or 0.0), 3)
            min_sec = float(transition.get("min_travel_time_sec", 0.0) or 0.0) if transition else None
            max_sec = float(transition.get("max_travel_time_sec", 0.0) or 0.0) if transition else None
            observations.append(
                {
                    "identity_id": identity_id,
                    "src_camera_id": left.get("camera_id", ""),
                    "dst_camera_id": right.get("camera_id", ""),
                    "src_event_id": left.get("event_id", ""),
                    "dst_event_id": right.get("event_id", ""),
                    "observed_delta_sec": delta,
                    "expected_min_sec": min_sec,
                    "expected_max_sec": max_sec,
                    "within_window": bool(
                        transition and min_sec is not None and max_sec is not None and min_sec <= delta <= max_sec
                    ),
                    "transition_rule_id": transition.get("transition_id", "") if transition else "",
                }
            )
    return observations


def main():
    parser = argparse.ArgumentParser(description="Compare observed unknown handoff deltas against the transition map.")
    parser.add_argument("--run-output-root", required=True)
    parser.add_argument(
        "--transition-map-config",
        default="insightface_demo_assets/runtime/config/camera_transition_map.new_dataset_demo.yaml",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    run_output_root = Path(args.run_output_root).resolve()
    transition_map_path = Path(args.transition_map_config).resolve()
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else run_output_root / "summaries" / "observed_travel_time_audit.json"
    )

    rows, resolved_events_path = load_resolved_rows(run_output_root)
    transitions = load_transition_map(transition_map_path)
    observations = observed_deltas(rows, transitions)
    payload = {
        "run_output_root": str(run_output_root),
        "resolved_events_path": str(resolved_events_path),
        "transition_map_config": str(transition_map_path),
        "observed_handoffs": observations,
        "within_window_count": sum(1 for row in observations if row.get("within_window")),
        "out_of_window_count": sum(1 for row in observations if not row.get("within_window")),
    }
    save_json(output_json, payload)
    print(output_json)


if __name__ == "__main__":
    main()
