import argparse
import json
from pathlib import Path

from run_face_resolution_demo import main as run_face_resolution_main


def _rewrite_runtime_paths(runtime_config, output_root: Path):
    runtime_dir = output_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    updated = dict(runtime_config)
    updated["resolved_events_csv"] = str(runtime_dir / "resolved_events_template.csv")
    updated["unknown_profiles_csv"] = str(runtime_dir / "unknown_profiles_template.csv")
    updated["known_face_embeddings_csv"] = str(runtime_dir / "known_face_embeddings_template.csv")
    return updated


def main():
    parser = argparse.ArgumentParser(description="Rerun face/body/association resolution on existing stage inputs with a policy variant.")
    parser.add_argument("--base-runtime-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--association-policy-config", default="")
    parser.add_argument("--known-face-manifest-csv", default="")
    parser.add_argument("--known-face-gallery-root", default="")
    args = parser.parse_args()

    base_runtime_config = Path(args.base_runtime_config).resolve()
    output_root = Path(args.output_root).resolve()
    runtime_config = json.loads(base_runtime_config.read_text(encoding="utf-8"))
    runtime_config = _rewrite_runtime_paths(runtime_config, output_root)
    if args.association_policy_config:
        runtime_config["association_policy_config"] = str(Path(args.association_policy_config).resolve())
    if args.known_face_manifest_csv:
        runtime_config["known_face_manifest_csv"] = str(Path(args.known_face_manifest_csv).resolve())
    if args.known_face_gallery_root:
        runtime_config["known_face_gallery_root"] = str(Path(args.known_face_gallery_root).resolve())
    runtime_config_path = output_root / "runtime" / "face_demo_runtime_config.json"
    runtime_config_path.write_text(json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8")
    run_face_resolution_main(runtime_config_path)
    print(runtime_config_path)


if __name__ == "__main__":
    main()
