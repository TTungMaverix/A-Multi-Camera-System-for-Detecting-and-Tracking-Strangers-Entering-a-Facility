import argparse
import csv
import json
from pathlib import Path

import yaml

from offline_pipeline.direction_logic import evaluate_direction


def _default_cases():
    return [
        {
            "case_id": "strong_in",
            "expected": "IN",
            "points": [{"x": 50, "y": 10}, {"x": 50, "y": 30}, {"x": 50, "y": 55}, {"x": 50, "y": 85}],
            "spatial_history": [],
        },
        {
            "case_id": "strong_out",
            "expected": "OUT",
            "points": [{"x": 50, "y": 88}, {"x": 50, "y": 72}, {"x": 50, "y": 45}, {"x": 50, "y": 15}],
            "spatial_history": [],
        },
        {
            "case_id": "unstable_none",
            "expected": "NONE",
            "points": [{"x": 50, "y": 46}, {"x": 50, "y": 49}, {"x": 50, "y": 51}, {"x": 50, "y": 53}],
            "spatial_history": [],
        },
        {
            "case_id": "zone_transition_required_missing",
            "expected": "NONE",
            "points": [{"x": 50, "y": 10}, {"x": 50, "y": 30}, {"x": 50, "y": 55}, {"x": 50, "y": 85}],
            "spatial_history": [
                {"zone_type": "entry"},
                {"zone_type": "entry"},
                {"zone_type": "entry"},
                {"zone_type": "entry"},
            ],
            "require_zone_transition": True,
        },
        {
            "case_id": "zone_transition_required_present",
            "expected": "IN",
            "points": [{"x": 50, "y": 10}, {"x": 50, "y": 30}, {"x": 50, "y": 55}, {"x": 50, "y": 85}],
            "spatial_history": [
                {"zone_type": "entry"},
                {"zone_type": "entry"},
                {"zone_type": "interior"},
                {"zone_type": "interior"},
            ],
            "require_zone_transition": True,
        },
    ]


def load_direction_config(calibration_path: Path):
    payload = yaml.safe_load(calibration_path.read_text(encoding="utf-8")) or {}
    calibration = payload.get("scene_calibration", payload)
    return calibration.get("direction_filter", {})


def main():
    parser = argparse.ArgumentParser(description="Run an independent direction-validation check.")
    parser.add_argument(
        "--scene-calibration-config",
        default="insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml",
    )
    parser.add_argument("--output-root", default="outputs/evaluations/direction_validation")
    args = parser.parse_args()

    calibration_path = Path(args.scene_calibration_config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    direction_cfg = load_direction_config(calibration_path)
    line = [[0, 50], [100, 50]]
    in_side_point = [50, 80]
    case_rows = []

    for case in _default_cases():
        cfg = dict(direction_cfg)
        if case.get("require_zone_transition") is not None:
            cfg["require_zone_transition"] = bool(case["require_zone_transition"])
        result = evaluate_direction(
            case["points"],
            line,
            in_side_point,
            spatial_history=case.get("spatial_history", []),
            config=cfg,
        )
        case_rows.append(
            {
                "case_id": case["case_id"],
                "expected": case["expected"],
                "predicted": result["decision"],
                "passed": result["decision"] == case["expected"],
                "reason": result.get("reason", ""),
                "momentum_px": result.get("momentum_px", 0.0),
                "inside_ratio": result.get("inside_ratio", 0.0),
                "zone_transition_ok": result.get("zone_transition_ok", False),
            }
        )

    passed = sum(1 for row in case_rows if row["passed"])
    summary = {
        "scene_calibration_config": str(calibration_path.resolve()),
        "case_count": len(case_rows),
        "passed_case_count": passed,
        "direction_accuracy": round(passed / max(len(case_rows), 1), 4),
        "direction_config_used": direction_cfg,
        "cases": case_rows,
    }
    (output_root / "direction_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_root / "direction_validation_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)
    print(f"DIRECTION_ACCURACY={summary['direction_accuracy']}")
    print(output_root / "direction_validation_summary.json")


if __name__ == "__main__":
    main()
