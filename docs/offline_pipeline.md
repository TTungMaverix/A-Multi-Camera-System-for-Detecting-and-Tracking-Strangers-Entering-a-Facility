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
- quality-aware body tracklet pooling as the active policy
- camera-role-aware face capture with strict best-shot gating
- topology-supported sequential accept for near-threshold body-only cases
- per-clip evaluation that now separates appearance-only passes from topology-supported passes
- a2 pair-level overlay debugging for direction/event failures
- a3 traditional-CV preprocessing and bbox-shrink benchmarking

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
- `association_policy.body_reid.extractor_name`
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
5. post-filter detections by ROI coverage and anchor point
6. build track rows
7. build direction-filtered entry events
8. select best body/head crops
9. run known/unknown resolution
10. apply topology hard filter before face/body scoring
11. emit resolved events, mappings, timelines, and handoff summaries

For the current phase, step 8 and step 10 are more structured than before:

- body evidence is pooled from the best valid body crops in the same tracklet
- the active policy uses `tracklet_pooling_mode = quality_aware`
- face evidence is only attempted on cameras whose `face_capture_mode` allows it
- sequential reuse can still succeed when body evidence is slightly below the global sequential threshold, but only if topology/time/zone/subzone support is strong and explicit

For the New Dataset profile, the active offline demo path first builds a **logical manifest** from clip stems shared across the 2 physical camera folders, then expands them into 4 logical streams with explicit time offsets.

ROI is not metadata in this phase. Boxes outside the configured processing polygon are killed before ReID and association.

## Current Demo Assumptions

The current New Dataset assumptions are:

- physical camera folders: `Camera 1`, `Camera 2`
- pairing rule: shared stem (`a1`, `a2`, `a3`, `b1`, ...)
- calibration reuse rule: same source camera -> same calibration
- physical travel time `Camera 1 -> Camera 2`: configured under the active transition map
- logical demo timeline:
  - `C1 @ 0s`
  - `C2 @ 10s`
  - `C3 @ 20s`
  - `C4 @ 30s`

Anchor-point defaults:

- `C1`, `C3`: `bottom_center`
- `C2`, `C4`: `center_center`

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

Dataset inventory + calibration reuse audit:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_inventory.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --output-dir "outputs/evaluations/a2_a3_cv_phase_inventory"
```

Full per-clip evaluation across all currently paired local clips:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_evaluation.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --inventory-json ".\outputs\evaluations\a2_a3_cv_phase_inventory\dataset_inventory.json" --calibration-reuse-json ".\outputs\evaluations\a2_a3_cv_phase_inventory\calibration_reuse_summary.json" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current" --baseline-output-dir ".\outputs\evaluations\new_dataset_quality_pooling_phase_current"
```

Independent direction validation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_direction_validation.py" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --output-root "outputs/evaluations/direction_validation_tracklet_phase"
```

Body tracklet comparison on an existing run root:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_body_tracklet_evaluation.py" --run-output-root "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4" --output-dir "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet_phase_current"
```

a2 overlay debug:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_pair_debug.py" --pipeline-config ".\outputs\evaluations\a2_a3_cv_phase_current\tmp_phase_pipeline.yaml" --run-output-root ".\outputs\evaluations\a2_a3_cv_phase_current\offline_runs\a2" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current\a2_debug" --pair-id a2
```

a3 hard-case crop/preprocessing benchmark:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_a3_hard_case_analysis.py" --run-output-root ".\outputs\evaluations\a2_a3_cv_phase_current\offline_runs\a3" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current\a3_hard_case" --pair-id a3 --association-policy-config ".\insightface_demo_assets\runtime\config\association_policy.new_dataset_demo.yaml"
```

## Current Limitations

- local paired coverage is now `a1`, `a2`, `a3`, and `b1`, but the target of at least `5` paired clips is still unmet
- all current clips can reuse the existing source-camera calibration; no per-clip redraw is needed right now
- `a2` no longer fails at event creation; it now emits `4` entry events after the late-start inside-entry fallback, but still does not reuse cross-camera IDs
- `a3` remains the hardest current case and still fails cross-camera reuse
- the best traditional-CV combo on `a3` is currently `gray_world + bbox_shrink_ratio=0.1`, which raises the hard-case body score but still leaves it below `0.72`
- face embeddings are still rare; only `b1` produced a created face embedding in the current evaluation sweep, and even that did not become a decisive identity anchor
- quality-aware pooling is more principled than mean-only pooling, but it does not yet push the physical `C1 -> C2` same-identity body scores above `0.72`
- logical `C3/C4` remain demo expansions from the 2 physical cameras
