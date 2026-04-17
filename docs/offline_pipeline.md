# Offline End-to-End Pipeline

## Current Phase

The current offline debug path is the **Wildtrack 4-camera ROI benchmark**.

It uses:

- four different recorded videos: `C3`, `C5`, `C6`, `C7`
- real `YOLOv8n + ByteTrack` inference per camera
- short-gap tracklet linking after ByteTrack
- calibrated ROI masking before downstream identity logic
- inward-direction filtering before event creation
- face-first matching with pose-aware best-shot selection
- OSNet body ReID fallback
- topology/travel-time hard filtering before similarity

The earlier single-source sequential replay proof still exists, but it is no longer the primary benchmark of the current phase.

## Entry Points

Current 4-camera ROI benchmark:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_wildtrack_4cam_roi_benchmark.ps1"
```

Direct orchestrator wrapper:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_offline_multicam_pipeline.ps1"
```

Direct Python:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.wildtrack_4cam_inference_roi_benchmark.yaml"
```

Paired no-ROI baseline:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.wildtrack_4cam_inference_no_roi_benchmark.yaml"
```

## Main Configs

Primary 4-camera configs:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.wildtrack_4cam_inference_roi_benchmark.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.wildtrack_4cam_inference_no_roi_benchmark.yaml`
- `insightface_demo_assets/runtime/config/manual_scene_calibration.wildtrack_4cam_phase.yaml`
- `insightface_demo_assets/runtime/config/camera_transition_map.wildtrack_4cam_phase.yaml`
- `insightface_demo_assets/runtime/config/association_policy.wildtrack_phase_f_tuned.yaml`

Important config blocks in the current phase:

- `multi_source_inference.camera_ids`
- `multi_source_inference.tracklet_linking`
- `multi_source_inference.roi_filter`
- `multi_source_inference.cache`
- `scene_calibration_config`
- `camera_transition_map_config`
- `association_policy_config`

## Backend Flow

For each configured camera video:

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

ROI is not metadata in this phase. Boxes outside the configured processing polygon are killed before ReID and association.

## Current Benchmark Pair

The current short benchmark window is:

- frame range: `0..360`
- actual duration: `6.006s`
- stride: `12`
- cameras: `C3`, `C5`, `C6`, `C7`

Official phase numbers:

- no ROI: `FP=1805`, `FN=249`, `MOTA=-6.3238`
- ROI: `FP=526`, `FN=250`, `MOTA=-1.977`

Interpretation:

- ROI masking clearly reduces false positives
- MOTA is still negative, so the local detect+track stage remains the main bottleneck

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
- `evaluation/quantitative_metrics_summary.json`

## Sequential Replay Reference Path

The earlier single-source replay proof remains available when a simpler, controlled sanity run is needed:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_50s.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_90s.yaml`
