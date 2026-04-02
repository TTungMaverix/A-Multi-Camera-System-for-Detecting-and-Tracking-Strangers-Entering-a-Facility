from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EVENT_FIELDNAMES = [
    "event_id",
    "timestamp",
    "camera_id",
    "local_track_id",
    "global_id",
    "identity_type",
    "direction",
    "known_match_score",
    "association_score",
    "snapshot_path",
    "predicted_next_cameras_json",
]


def append_event_csv(csv_path: str | Path, event: dict[str, Any]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = dict(event)
    value = row.get("predicted_next_cameras_json")
    if isinstance(value, (dict, list)):
        row["predicted_next_cameras_json"] = json.dumps(value, ensure_ascii=False)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in EVENT_FIELDNAMES})


def write_runtime_summary(summary_path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
