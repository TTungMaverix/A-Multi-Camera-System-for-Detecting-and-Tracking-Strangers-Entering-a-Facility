# Offline End-to-End Pipeline

## Current Backend

The default offline runnable pipeline is now the **single-source sequential replay inference proof phase**.

- one physical source video: `Wildtrack/cam6.mp4`
- four virtual camera passes: `C1`, `C2`, `C3`, `C4`
- fake timestamp offsets between passes to simulate travel time
- per-camera track extraction: real `YOLOv8n + ByteTrack` inference on `Wildtrack/cam6.mp4`
- direction filter: manual scene calibration + trajectory/momentum logic from `insightface_demo_assets/runtime/config/manual_scene_calibration.single_source_sequential_c6.json`
- face matching + known/unknown + cross-camera association: `insightface_demo_assets/runtime/run_face_resolution_demo.py`

The older Wildtrack 4-source mode still exists as a reference backend, but it is not the default rescue-demo path.

## Entry Point

Main offline command:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_multicam_identity_demo.ps1"
```

Direct video-phase command:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_single_source_sequential_video_phase.ps1"
```

Direct Python:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.single_source_sequential_c6_inference_50s.yaml"
```

The generic orchestrator wrapper still exists, but the default defense/demo path now resolves to the single-source sequential replay config.

## Config

Main verified inference video-phase config:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_50s.yaml`

Longer stress config that currently exceeds the local verification timeout:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_90s.yaml`

Earlier short sanity config:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6.yaml`

Reference Wildtrack multi-source config:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.example.yaml`

Low-load config:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.low_load.yaml`

Important fields:

- `dataset.video_sources`: current input source paths for the run
- `single_source_replay.source_camera_id`
- `single_source_replay.source_video`
- `single_source_replay.replay_count`
- `single_source_replay.virtual_camera_ids`
- `single_source_replay.virtual_time_offsets_sec`
- `single_source_replay.virtual_frame_offsets`
- `wildtrack_demo_config`: dataset-specific ROI, line crossing, best-shot window
- `scene_calibration_config`: required manual ROI/zone/subzone calibration
- `wildtrack_demo_config.best_shot_selection`: line-aware best-shot policy
- `known_gallery.manifest_csv`
- `known_gallery.gallery_root`
- `association_policy_config`
- `camera_transition_map_config`
- `execution.mode`
- `low_load`

For the rescue demo, the most important extra files are:

- `wildtrack_demo/single_source_sequential_c6_demo_config.json`
- `insightface_demo_assets/runtime/config/camera_transition_map.single_source_sequential_c6.yaml`
- `insightface_demo_assets/runtime/config/manual_scene_calibration.single_source_sequential_c6.json`

## Event Creation Conditions

Only `direction=IN` entry events go to face matching.

Minimum event packet fields in the offline flow:

- `camera_id`
- `relative_sec`
- `local_track_id`
- `global_gt_id` for audit only
- `bbox_*`
- `best_head_crop`
- `best_body_crop`
- `direction`
- `zone_id`
- `subzone_id`

Event creation rules in the current backend:

1. run YOLO person detection plus ByteTrack on the single source video
2. project the inferred source tracks into virtual cameras with configured fake frame and time offsets
3. keep rows whose foot point stays inside the configured ROI
4. for entry cameras, use calibrated tripwire + motion history + inward momentum to stabilize `IN`
5. only after line crossing, create `ENTRY_IN`
6. select a best shot inside the configured frame window
7. when enabled, prefer post-anchor frames in higher-priority subzones
8. create best-shot body/head crops
9. send that event to face matching and then to cross-camera association

## Output Layout

For each offline run:

- `outputs/offline_runs/<run_name>/tracks/`
- `outputs/offline_runs/<run_name>/events/`
- `outputs/offline_runs/<run_name>/crops/`
- `outputs/offline_runs/<run_name>/timelines/`
- `outputs/offline_runs/<run_name>/summaries/`
- `outputs/offline_runs/<run_name>/audit/`
- `outputs/offline_runs/<run_name>/association_logs/`
- `outputs/offline_runs/<run_name>/runtime/`

Main final files:

- `events/resolved_events.csv`
- `events/unknown_id_mapping.csv`
- `timelines/stream_identity_timeline.csv`
- `summaries/face_resolution_summary.json`
- `summaries/single_source_replay_manifest.json`
- `association_logs/association_decisions.jsonl`
- `audit/audit_report.md`
- `audit/entry_event_assignment_audit.csv`
- `audit/audit_event_generation_subzones.csv`

The audit CSV files include line-aware best-shot metadata so event creation and unknown ID reuse can be debugged without rerunning the full pipeline. The summaries directory now also exports `face_body_usage_summary.json` for video-phase face-vs-body evidence.
