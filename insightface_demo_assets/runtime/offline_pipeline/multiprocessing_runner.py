import json
import multiprocessing as mp
import traceback
from pathlib import Path

import yaml

from association_core import load_camera_transition_map
from offline_pipeline.event_builder import (
    build_entry_anchor_packets,
    extract_camera_track_rows,
    fieldnames_for_rows,
    load_json,
    materialize_stage_inputs,
    save_json,
    write_csv,
)
from run_face_resolution_demo import main as run_face_resolution_main


def load_pipeline_config(config_path: Path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "offline_pipeline" in payload:
        return payload["offline_pipeline"]
    return payload


def resolve_path(project_root: Path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def build_face_runtime_config(offline_config, stage_inputs, output_root: Path):
    project_root = Path(stage_inputs["project_root"]).resolve()
    face_demo_config_path = resolve_path(project_root, offline_config["face_demo_config"])
    face_demo_config = load_json(face_demo_config_path)
    runtime_dir = output_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = dict(face_demo_config)
    runtime_config.update(
        {
            "project_root": str(project_root),
            "wildtrack_config_path": stage_inputs["wildtrack_config_path"],
            "tracks_csv": stage_inputs["tracks_csv"],
            "wildtrack_identity_queue_csv": stage_inputs["identity_queue_csv"],
            "resolved_events_csv": str(runtime_dir / "resolved_events_template.csv"),
            "unknown_profiles_csv": str(runtime_dir / "unknown_profiles_template.csv"),
            "known_face_embeddings_csv": str(runtime_dir / "known_face_embeddings_template.csv"),
        }
    )
    if offline_config.get("association_policy_config"):
        runtime_config["association_policy_config"] = str(
            resolve_path(project_root, offline_config["association_policy_config"])
        )
    if offline_config.get("camera_transition_map_config"):
        runtime_config["camera_transition_map_config"] = str(
            resolve_path(project_root, offline_config["camera_transition_map_config"])
        )
    if offline_config.get("known_gallery", {}).get("manifest_csv"):
        runtime_config["known_face_manifest_csv"] = str(
            resolve_path(project_root, offline_config["known_gallery"]["manifest_csv"])
        )
    if offline_config.get("known_gallery", {}).get("gallery_root"):
        runtime_config["known_face_gallery_root"] = str(
            resolve_path(project_root, offline_config["known_gallery"]["gallery_root"])
        )
    runtime_config_path = runtime_dir / "face_demo_runtime_config.json"
    runtime_config_path.write_text(json.dumps(runtime_config, ensure_ascii=False, indent=2), encoding="utf-8")
    return runtime_config_path


def sync_final_outputs(output_root: Path):
    runtime_dir = output_root / "runtime"
    events_dir = output_root / "events"
    timelines_dir = output_root / "timelines"
    summaries_dir = output_root / "summaries"
    audit_dir = output_root / "audit"
    association_logs_dir = output_root / "association_logs"
    for directory in (events_dir, timelines_dir, summaries_dir, audit_dir, association_logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copy_pairs = [
        (runtime_dir / "resolved_events_template.csv", events_dir / "resolved_events.csv"),
        (runtime_dir / "stream_identity_timeline.csv", timelines_dir / "stream_identity_timeline.csv"),
        (runtime_dir / "unknown_profiles_template.csv", timelines_dir / "unknown_profiles.csv"),
        (runtime_dir / "face_resolution_summary.json", summaries_dir / "face_resolution_summary.json"),
    ]
    for src, dst in copy_pairs:
        if src.exists():
            dst.write_bytes(src.read_bytes())

    for audit_file in runtime_dir.glob("audit_*"):
        if audit_file.is_file():
            (audit_dir / audit_file.name).write_bytes(audit_file.read_bytes())
    runtime_association_dir = runtime_dir / "association_logs"
    if runtime_association_dir.exists():
        for path in runtime_association_dir.iterdir():
            if path.is_file():
                (association_logs_dir / path.name).write_bytes(path.read_bytes())


def _put_batch(queue, packet_type, camera_id, rows, batch_size):
    for index in range(0, len(rows), batch_size):
        queue.put(
            {
                "packet_type": packet_type,
                "camera_id": camera_id,
                "rows": rows[index : index + batch_size],
            }
        )


def camera_worker(worker_context, queue):
    camera_id = worker_context["camera_id"]
    try:
        queue.put({"packet_type": "worker_started", "camera_id": camera_id})
        rows = extract_camera_track_rows(
            Path(worker_context["dataset_root"]),
            camera_id,
            worker_context["camera_cfg"],
            worker_context["fps"],
            Path(worker_context["video_source"]),
            worker_context["low_load_cfg"],
            worker_context["frame_source_mode"],
        )
        _put_batch(queue, "track_row_batch", camera_id, rows, worker_context["packet_batch_size"])
        entry_anchor_packets = []
        if worker_context["camera_cfg"].get("role") == "entry":
            entry_anchor_packets = build_entry_anchor_packets(
                camera_id,
                worker_context["camera_cfg"],
                rows,
                worker_context["line_threshold"],
                worker_context["transition_map"],
            )
            _put_batch(queue, "entry_anchor_batch", camera_id, entry_anchor_packets, worker_context["packet_batch_size"])
        queue.put(
            {
                "packet_type": "camera_summary",
                "camera_id": camera_id,
                "track_row_count": len(rows),
                "entry_anchor_count": len(entry_anchor_packets),
            }
        )
    except Exception as exc:
        queue.put(
            {
                "packet_type": "worker_error",
                "camera_id": camera_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        queue.put({"packet_type": "worker_done", "camera_id": camera_id})


def run_offline_pipeline_multiprocess(config_path: Path, cli_overrides=None):
    cli_overrides = cli_overrides or {}
    offline_config = load_pipeline_config(config_path)
    project_root = resolve_path(config_path.parent, offline_config.get("project_root", "."))
    offline_config["project_root"] = str(project_root)
    if cli_overrides.get("association_policy_config"):
        offline_config["association_policy_config"] = cli_overrides["association_policy_config"]
    if cli_overrides.get("camera_transition_map_config"):
        offline_config["camera_transition_map_config"] = cli_overrides["camera_transition_map_config"]
    if cli_overrides.get("known_manifest_csv"):
        offline_config.setdefault("known_gallery", {})
        offline_config["known_gallery"]["manifest_csv"] = cli_overrides["known_manifest_csv"]
    if cli_overrides.get("known_gallery_root"):
        offline_config.setdefault("known_gallery", {})
        offline_config["known_gallery"]["gallery_root"] = cli_overrides["known_gallery_root"]
    if cli_overrides.get("video_sources"):
        offline_config.setdefault("dataset", {})
        offline_config["dataset"]["video_sources"] = cli_overrides["video_sources"]

    output_root = resolve_path(project_root, offline_config["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    wildtrack_config_path = resolve_path(project_root, offline_config["wildtrack_demo_config"])
    wildtrack_config = load_json(wildtrack_config_path)
    transition_map, transition_runtime = load_camera_transition_map(
        wildtrack_config,
        config_path=offline_config.get("camera_transition_map_config", ""),
        base_dir=config_path.parent,
    )
    dataset_root = resolve_path(project_root, offline_config["dataset"]["root"])
    frame_source_mode = offline_config.get("frame_source_mode", "video_files")
    low_load_cfg = offline_config.get("low_load", {})
    execution_cfg = offline_config.get("execution", {})
    packet_batch_size = max(1, int(execution_cfg.get("packet_batch_size", 256)))
    queue_max_size = max(8, int(execution_cfg.get("queue_max_size", 64)))
    start_method = execution_cfg.get("start_method", "spawn")
    try:
        mp.set_start_method(start_method, force=True)
    except RuntimeError:
        pass

    queue = mp.Queue(maxsize=queue_max_size)
    workers = []
    event_log_rows = []
    track_rows_by_camera = {camera_id: [] for camera_id in wildtrack_config["selected_cameras"]}
    entry_anchor_packets = []
    worker_summaries = {}
    errors = []

    for camera_id in wildtrack_config["selected_cameras"]:
        video_source = resolve_path(project_root, offline_config["dataset"]["video_sources"][camera_id])
        worker_context = {
            "camera_id": camera_id,
            "dataset_root": str(dataset_root),
            "camera_cfg": wildtrack_config["cameras"][camera_id],
            "fps": float(wildtrack_config["assumed_video_fps"]),
            "video_source": str(video_source),
            "low_load_cfg": low_load_cfg,
            "frame_source_mode": frame_source_mode,
            "line_threshold": float(wildtrack_config["line_crossing_distance_threshold"]),
            "transition_map": transition_map,
            "packet_batch_size": packet_batch_size,
        }
        process = mp.Process(target=camera_worker, args=(worker_context, queue), name=f"edge_worker_{camera_id}")
        process.start()
        workers.append(process)

    done_workers = set()
    packet_counts = {
        "total_packets_emitted": 0,
        "track_row_packets": 0,
        "entry_anchor_packets": 0,
        "camera_summary_packets": 0,
        "worker_error_packets": 0,
    }
    while len(done_workers) < len(workers):
        packet = queue.get()
        packet_counts["total_packets_emitted"] += 1
        packet_type = packet["packet_type"]
        event_log_rows.append(packet)
        camera_id = packet.get("camera_id", "")
        if packet_type == "track_row_batch":
            packet_counts["track_row_packets"] += 1
            track_rows_by_camera[camera_id].extend(packet["rows"])
        elif packet_type == "entry_anchor_batch":
            packet_counts["entry_anchor_packets"] += 1
            entry_anchor_packets.extend(packet["rows"])
        elif packet_type == "camera_summary":
            packet_counts["camera_summary_packets"] += 1
            worker_summaries[camera_id] = {
                "track_row_count": packet["track_row_count"],
                "entry_anchor_count": packet["entry_anchor_count"],
            }
        elif packet_type == "worker_error":
            packet_counts["worker_error_packets"] += 1
            errors.append(packet)
        elif packet_type == "worker_done":
            done_workers.add(camera_id)

    for process in workers:
        process.join()

    stage_inputs = materialize_stage_inputs(
        project_root,
        dataset_root,
        wildtrack_config_path,
        wildtrack_config,
        output_root,
        track_rows_by_camera,
        frame_source_mode,
        low_load_cfg,
        transition_map,
    )
    stage_inputs["video_sources"] = {
        camera_id: str(resolve_path(project_root, offline_config["dataset"]["video_sources"][camera_id]))
        for camera_id in wildtrack_config["selected_cameras"]
    }
    stage_inputs["edge_entry_anchor_count"] = len(entry_anchor_packets)

    write_csv(
        logs_dir / "edge_entry_anchor_packets.csv",
        entry_anchor_packets,
        fieldnames_for_rows(entry_anchor_packets, ["packet_type", "camera_id"]) if entry_anchor_packets else ["packet_type", "camera_id"],
    )
    save_json(logs_dir / "multiprocessing_events.json", event_log_rows)
    multiproc_summary = {
        "execution_mode": "multiprocessing",
        "worker_count": len(workers),
        "packet_counts": packet_counts,
        "worker_summaries": worker_summaries,
        "error_count": len(errors),
        "errors": errors,
    }
    save_json(logs_dir / "multiprocessing_summary.json", multiproc_summary)

    runtime_config_path = build_face_runtime_config(offline_config, stage_inputs, output_root)
    run_face_resolution_main(runtime_config_path)
    sync_final_outputs(output_root)
    pipeline_summary = {
        "pipeline_name": offline_config.get("pipeline_name", "offline_multicam_pipeline"),
        "source_backend": offline_config.get("source_backend", "wildtrack_gt_annotations"),
        "execution_mode": "multiprocessing",
        "config_path": str(config_path),
        "output_root": str(output_root),
        "stage_inputs": stage_inputs,
        "runtime_config_path": str(runtime_config_path),
        "camera_transition_map_runtime": transition_runtime,
        "multiprocessing_summary": multiproc_summary,
        "final_outputs": {
            "resolved_events_csv": str(output_root / "events" / "resolved_events.csv"),
            "stream_identity_timeline_csv": str(output_root / "timelines" / "stream_identity_timeline.csv"),
            "summary_json": str(output_root / "summaries" / "face_resolution_summary.json"),
            "association_logs_dir": str(output_root / "association_logs"),
            "audit_dir": str(output_root / "audit"),
        },
    }
    save_json(output_root / "summaries" / "offline_pipeline_summary.json", pipeline_summary)
    return pipeline_summary
