# Offline End-to-End Pipeline

## Current Backend

The current offline runnable pipeline is **video-file-first**, but the detect/track stage is still a **GT-backed Wildtrack provider**:

- video sources: `Wildtrack/cam3.mp4`, `cam5.mp4`, `cam6.mp4`, `cam7.mp4`
- per-camera track extraction: `Wildtrack/annotations_positions/*.json`
- direction filter: entry-camera line crossing from `wildtrack_demo/wildtrack_demo_config.json`
- face matching + known/unknown + cross-camera association: `insightface_demo_assets/runtime/run_face_resolution_demo.py`

This keeps the thesis demo honest:

- there is now one offline orchestration flow
- direction filtering and event creation are in that flow
- the current tracking backend is still annotation-backed, not a heavy detector+tracker inference stack

## Entry Point

Main offline command:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
powershell -ExecutionPolicy Bypass -File ".\run_offline_multicam_pipeline.ps1"
```

Compatibility command:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
powershell -ExecutionPolicy Bypass -File ".\run_multicam_identity_demo.ps1"
```

Both currently resolve to the same offline orchestrator.

## Config

Main config:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.example.yaml`

Low-load config:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.low_load.yaml`

Important fields:

- `dataset.video_sources`: 4 input videos
- `wildtrack_demo_config`: dataset-specific ROI, line crossing, best-shot window
- `known_gallery.manifest_csv`
- `known_gallery.gallery_root`
- `association_policy_config`
- `camera_transition_map_config`
- `execution.mode`
- `low_load`

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

1. keep rows whose foot point stays inside the configured ROI
2. for entry cameras, detect line crossing into the protected side
3. only after line crossing, create `ENTRY_IN`
4. create best-shot body/head crops
5. send that event to face matching and then to cross-camera association

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
- `timelines/stream_identity_timeline.csv`
- `summaries/face_resolution_summary.json`
- `association_logs/association_decisions.jsonl`
- `audit/audit_report.md`
