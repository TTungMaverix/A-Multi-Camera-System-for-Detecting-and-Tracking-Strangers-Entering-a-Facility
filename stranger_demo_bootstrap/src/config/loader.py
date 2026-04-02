from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_output_dirs(app_config: dict[str, Any]) -> None:
    storage = app_config.get("storage", {})
    for key in ("event_dir", "snapshot_dir", "clip_dir", "log_dir"):
        value = storage.get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)
    Path("data/outputs/debug").mkdir(parents=True, exist_ok=True)
    Path("data/db").mkdir(parents=True, exist_ok=True)
