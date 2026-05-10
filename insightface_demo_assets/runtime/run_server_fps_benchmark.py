import argparse
import json
import time
from pathlib import Path

from scene_calibration import build_runtime_camera_calibration, load_scene_calibration
from run_live_event_demo_server import FrameBufferHub, OfflineReplayFrameWorker


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark offline replay frame workers without serving HTTP.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Offline run output root with tracks/events artifacts.",
    )
    parser.add_argument(
        "--scene-calibration-config",
        default="insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml",
        help="Scene calibration YAML used by the live demo server.",
    )
    parser.add_argument("--duration-sec", type=float, default=6.0)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument(
        "--summary-json",
        default="outputs/evaluations/server_fps_benchmark/server_runtime_summary.json",
        help="Path to write benchmark summary JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    calibration_path = (project_root / args.scene_calibration_config).resolve()
    summary_path = (project_root / args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    calibration, _runtime = load_scene_calibration(
        config_path=str(calibration_path),
        base_dir=project_root,
        required=True,
    )
    hub = FrameBufferHub()
    workers = []
    for camera_id, camera_cfg in sorted((calibration.get("cameras", {}) or {}).items()):
        width = int(camera_cfg.get("frame_size", {}).get("width") or camera_cfg.get("width") or 0)
        height = int(camera_cfg.get("frame_size", {}).get("height") or camera_cfg.get("height") or 0)
        if not width or not height:
            width = 1280
            height = 720
        runtime_camera = build_runtime_camera_calibration(camera_cfg, width, height)
        worker = OfflineReplayFrameWorker(
            camera_id,
            camera_cfg,
            runtime_camera,
            project_root,
            output_root,
            hub,
            target_fps=args.target_fps,
        )
        workers.append(worker)

    started_at = time.time()
    for worker in workers:
        worker.start()
    time.sleep(max(1.0, float(args.duration_sec)))
    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join(timeout=3.0)
    elapsed_sec = time.time() - started_at
    snapshot = hub.snapshot()
    fps_values = [
        float(state.get("fps_estimate") or 0.0)
        for state in snapshot.values()
        if not state.get("error") and state.get("fps_estimate")
    ]
    payload = {
        "architecture": "offline_replay_frame_workers_with_latest_frame_buffer",
        "api_serving_path": "HTTP endpoints read latest JPEG/state from FrameBufferHub; workers own decode and overlay.",
        "target_fps": float(args.target_fps),
        "elapsed_sec": round(elapsed_sec, 3),
        "camera_count": len(workers),
        "camera_state": snapshot,
        "avg_worker_fps": round(sum(fps_values) / len(fps_values), 3) if fps_values else 0.0,
        "min_worker_fps": round(min(fps_values), 3) if fps_values else 0.0,
        "max_worker_fps": round(max(fps_values), 3) if fps_values else 0.0,
        "connection_abort_handling": "MJPEG endpoint catches BrokenPipeError, ConnectionAbortedError, and ConnectionResetError per client stream.",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
