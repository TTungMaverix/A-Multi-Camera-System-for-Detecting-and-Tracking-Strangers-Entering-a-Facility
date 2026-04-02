from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.main import build_runtime_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate configs and prepare demo runtime summary.")
    parser.add_argument("--app-config", default="config/app.yaml")
    parser.add_argument("--cameras-config", default="config/cameras.yaml")
    parser.add_argument("--topology-config", default="config/topology.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_runtime_summary(
        app_config_path=args.app_config,
        cameras_config_path=args.cameras_config,
        topology_config_path=args.topology_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
