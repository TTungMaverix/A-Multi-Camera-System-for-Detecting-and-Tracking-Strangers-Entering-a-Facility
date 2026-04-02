from __future__ import annotations

import json
import sys
from typing import Any

from src.config.loader import ensure_output_dirs, load_yaml
from src.core.event_logger import write_runtime_summary
from src.core.topology import candidate_next_cameras

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_runtime_summary(
    app_config_path: str = "config/app.yaml",
    cameras_config_path: str = "config/cameras.yaml",
    topology_config_path: str = "config/topology.yaml",
) -> dict[str, Any]:
    app_config = load_yaml(app_config_path)
    cameras_config = load_yaml(cameras_config_path)
    topology_config = load_yaml(topology_config_path)
    ensure_output_dirs(app_config)

    cameras = cameras_config.get("cameras", [])
    entry_cameras = [item["camera_id"] for item in cameras if item.get("role") == "entry"]
    summary = {
        "app_name": app_config.get("app_name", "stranger-demo"),
        "camera_count": len(cameras),
        "camera_ids": [item.get("camera_id") for item in cameras],
        "entry_cameras": entry_cameras,
        "visualization": app_config.get("visualization", {}),
        "predicted_next_from_entry": {
            camera_id: candidate_next_cameras(topology_config, camera_id) for camera_id in entry_cameras
        },
        "status": "bootstrap-ready",
    }
    write_runtime_summary("data/outputs/debug/runtime_summary.json", summary)
    return summary


def main() -> None:
    summary = build_runtime_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
