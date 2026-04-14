# Single-Source Sequential 4-Pass Sanity Demo

## Purpose

This mode exists to validate the core thesis pipeline in the easiest supervisor-approved scenario before returning to harder real multi-camera conditions.

The idea is:

1. take one source video from camera `C6`
2. replay it sequentially 4 times
3. label each replay as a different virtual camera: `C1`, `C2`, `C3`, `C4`
4. inject fake time offsets between passes to simulate travel time
5. verify that the same unknown person keeps the same `Unknown_Global_ID` across the 4 passes

If the same unknown person is still split into multiple unknown IDs in this mode, the core reuse logic is not ready for harder demo phases.

## Entry Point

Default command:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_multicam_identity_demo.ps1"
```

Equivalent direct command for the earlier short sanity slice:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_single_source_sequential_demo.ps1"
```

Current verified inference video-phase command:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_single_source_sequential_video_phase.ps1"
```

Direct Python:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.single_source_sequential_c6.yaml"
```

## Config Files

Main offline config:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6.yaml`

Virtual camera dataset/runtime config:

- `wildtrack_demo/single_source_sequential_c6_demo_config.json`

Virtual transition map:

- `insightface_demo_assets/runtime/config/camera_transition_map.single_source_sequential_c6.yaml`

Manual scene calibration used by all 4 replay passes:

- `insightface_demo_assets/runtime/config/manual_scene_calibration.single_source_sequential_c6.json`

## Important Config Fields

In `offline_pipeline_demo.single_source_sequential_c6.yaml`:

- `single_source_replay.source_camera_id`
- `single_source_replay.source_video`
- `single_source_replay.replay_count`
- `single_source_replay.virtual_camera_ids`
- `single_source_replay.virtual_time_offsets_sec`
- `single_source_replay.virtual_frame_offsets`
- `face_demo_overrides.demo_auto_enroll_count`
- `camera_transition_map_config`
- `scene_calibration_config`

Use these fields to change:

- which single video is replayed
- how many virtual passes are created
- which virtual camera IDs are used
- the fake travel-time offsets between passes

## Runtime Flow

The flow is:

1. load one real source stream from `C6`
2. run YOLO person detection plus ByteTrack on that source
3. clone the inferred source tracks into 4 virtual cameras
4. shift each virtual pass by configured fake timestamps and frame offsets
5. run the normal offline event-builder flow
6. keep only inward-direction entry events
7. create head/body crops
8. run face-first known matching
9. for unknowns, reuse `Unknown_Global_ID` through the existing association core
10. export events, timelines, mapping files, and association logs

## Expected Output

Main output folder:

- `outputs/offline_runs/single_source_sequential_c6_inference_phase_50s/`

Important files:

- `events/entry_in_events.csv`
- `events/identity_resolution_queue.csv`
- `events/resolved_events.csv`
- `events/unknown_id_mapping.csv`
- `timelines/stream_identity_timeline.csv`
- `timelines/unknown_profiles.csv`
- `summaries/face_resolution_summary.json`
- `summaries/single_source_replay_manifest.json`
- `association_logs/association_decisions.jsonl`
- `audit/`

## Pass Criterion

This sanity mode passes if the same unknown person from the single source video is assigned the same `Unknown_Global_ID` across the corresponding virtual passes `C1 -> C2 -> C3 -> C4`.
