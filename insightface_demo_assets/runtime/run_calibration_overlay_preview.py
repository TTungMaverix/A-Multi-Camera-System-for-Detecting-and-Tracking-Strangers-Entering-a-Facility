import argparse
from pathlib import Path

import cv2

from scene_calibration import draw_scene_overlay, load_runtime_scene_calibration, probe_frame_from_source


VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
CAMERA_FOLDER_MAP = {
    "C1": "Camera 1",
    "C2": "Camera 2",
    "C3": "Camera 1",
    "C4": "Camera 2",
}


def resolve_video_path(dataset_root: Path, demo_pair_id: str, camera_id: str):
    folder = CAMERA_FOLDER_MAP[camera_id]
    camera_dir = dataset_root / folder
    for extension in VIDEO_EXTENSIONS:
        direct = camera_dir / f"{demo_pair_id}{extension}"
        nested = camera_dir / demo_pair_id / f"{demo_pair_id}{extension}"
        if direct.exists():
            return direct
        if nested.exists():
            return nested
    raise RuntimeError(f"Missing preview video for {camera_id} pair={demo_pair_id} under {camera_dir}")


def write_png(path: Path, frame):
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(f"Failed to encode overlay image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def parse_args():
    parser = argparse.ArgumentParser(description="Render overlay previews for a manual scene calibration config.")
    parser.add_argument("--scene-calibration-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--demo-pair-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-ids", default="C1,C2")
    parser.add_argument("--frame-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.scene_calibration_config).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    camera_ids = [item.strip().upper() for item in str(args.camera_ids).split(",") if item.strip()]

    source_lookup = {}
    for camera_id in camera_ids:
        source_lookup[camera_id] = {
            "source_type": "file",
            "source": str(resolve_video_path(dataset_root, args.demo_pair_id, camera_id)),
        }
    _calibration, runtime_cameras, runtime_info = load_runtime_scene_calibration(
        config_path=str(config_path),
        base_dir=config_path.parent,
        camera_ids=camera_ids,
        source_lookup=source_lookup,
        required=True,
    )

    for camera_id in camera_ids:
        source_value = source_lookup[camera_id]["source"]
        frame = probe_frame_from_source("file", source_value, frame_idx=args.frame_index)
        overlay = draw_scene_overlay(frame, runtime_cameras[camera_id])
        cv2.putText(
            overlay,
            f"{camera_id} {overlay.shape[1]}x{overlay.shape[0]}",
            (24, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        output_path = output_dir / f"{camera_id}_overlay.png"
        write_png(output_path, overlay)
        print(f"OVERLAY_SAVED {camera_id} {output_path}")

    print(f"RUNTIME_SOURCE={runtime_info.get('source_path', '')}")


if __name__ == "__main__":
    main()
