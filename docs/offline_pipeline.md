# Offline End-to-End Pipeline

## Current Phase

The current active offline demo path is the **New Dataset logical 4-camera adapter**.

It uses:

- 2 physical videos from the self-recorded dataset
- a logical expansion into `C1 -> C2 -> C3 -> C4`
- real `YOLOv8n + ByteTrack` inference
- short-gap tracklet linking after ByteTrack
- calibrated ROI masking before downstream identity logic
- inward-direction filtering before event creation
- face-first matching with OSNet body fallback
- topology/travel-time hard filtering before similarity
- tracklet-pooled body evidence instead of one-shot body crops
- camera-role-aware face capture with strict best-shot gating
- topology-supported sequential accept for near-threshold body-only cases

The earlier Wildtrack benchmark path still remains in the repo for legacy comparison and regression checks, but it is no longer the active dataset narrative.

## Entry Points

Current New Dataset demo:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_new_dataset_logical_demo.ps1"
```

Direct orchestrator wrapper:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_offline_multicam_pipeline.ps1"
```

Direct Python:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml"
```

## Main Configs

Primary New Dataset configs:

- `insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml`
- `insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/camera_transition_map.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/bytetrack.new_dataset_demo.yaml`

Important config blocks in the current phase:

- `dataset_profile_config`
- `logical_demo.pair_id`
- `multi_source_inference.tracklet_linking`
- `multi_source_inference.roi_filter`
- `multi_source_inference.cache`
- `scene_calibration_config`
- `camera_transition_map_config`
- `association_policy_config`

Current phase-specific knobs that matter most:

- `association_policy.quality_gate.min_face_bbox_width / height / area`
- `association_policy.body_reid.tracklet_pooling_*`
- `association_policy.decision_policy.relation_thresholds.sequential.body_primary`
- `association_policy.decision_policy.topology_supported_accept`
- `dataset_profile.cameras[].face_capture_mode`

## Backend Flow

For each configured logical camera stream:

1. decode video frames
2. optionally pre-mask the frame with the calibrated ROI
3. run YOLO person detection and ByteTrack
4. run short-gap tracklet linking
5. post-filter detections by ROI coverage and foot-point
6. build track rows
7. build direction-filtered entry events
8. select best body/head crops
9. run known/unknown resolution
10. apply topology hard filter before face/body scoring
11. emit resolved events, mappings, timelines, and handoff summaries

For the current phase, step 8 and step 10 are more structured than before:

- body evidence is pooled from the best `5-10` valid body crops in the same tracklet
- face evidence is only attempted on cameras whose `face_capture_mode` allows it
- sequential reuse can still succeed when body evidence is slightly below the global
  sequential threshold, but only if topology/time/zone/subzone support is strong and explicit

For the New Dataset profile, the active offline demo path first builds a **logical manifest** from clip stems shared across the 2 physical camera folders, then expands them into 4 logical streams with explicit time offsets.

ROI is not metadata in this phase. Boxes outside the configured processing polygon are killed before ReID and association.

## Current Demo Assumptions

The current New Dataset assumptions are:

- physical camera folders: `Camera 1`, `Camera 2`
- pairing rule: shared stem (`a1`, `a2`, `b1`, ...)
- physical travel time `Camera 1 -> Camera 2`: about `10s`
- logical demo timeline:
  - `C1 @ 0s`
  - `C2 @ 10s`
  - `C3 @ 20s`
  - `C4 @ 30s`

Anchor-point defaults:

- `C1`, `C3`: `bottom_center`
- `C2`, `C4`: `center_center`

## Legacy Wildtrack Assets

Legacy Wildtrack configs and benchmark outputs are still present for audit and comparison, including:

- `offline_pipeline_demo.wildtrack_4cam_inference_roi_benchmark.yaml`
- `offline_pipeline_demo.wildtrack_4cam_inference_no_roi_benchmark.yaml`
- the earlier sequential replay proof configs

They are no longer the active default dataset path.

## Output Layout

Per run:

- `outputs/offline_runs/<run_name>/tracks/`
- `outputs/offline_runs/<run_name>/events/`
- `outputs/offline_runs/<run_name>/timelines/`
- `outputs/offline_runs/<run_name>/summaries/`
- `outputs/offline_runs/<run_name>/audit/`
- `outputs/offline_runs/<run_name>/association_logs/`
- `outputs/offline_runs/<run_name>/evaluation/`

Current phase artifacts that matter most:

- `events/resolved_events.csv`
- `events/latest_events.json`
- `timelines/unknown_identity_timeline.json`
- `summaries/cross_camera_handoff_summary.json`
- `summaries/stage_input_summary.json`
- `summaries/face_resolution_summary.json`
- `summaries/face_body_usage_summary.json`
- `runtime/association_logs/association_decisions.jsonl`

## Current Validation Commands

Full offline smoke with the active New Dataset config:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml"
```

Independent direction validation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_direction_validation.py" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --output-root "outputs/evaluations/direction_validation_tracklet_phase"
```

Body tracklet comparison on an existing run root:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_body_tracklet_evaluation.py" --run-output-root "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4" --output-dir "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet"
```

## Current Limitations

- only `a1` exists locally right now, so `a2` and `b1` cannot be validated yet
- face embeddings are still not created on the active `a1` smoke; the current phase makes
  those failures explicit instead of silently forcing bad embeddings
- logical `C3/C4` remain demo expansions from the 2 physical cameras
