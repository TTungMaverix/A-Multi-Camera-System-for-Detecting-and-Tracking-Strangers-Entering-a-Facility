# A Multi-Camera System for Detecting and Tracking Strangers Entering a Facility

Graduation project repository for a multi-camera security pipeline that stays within the original scope:

- 4 camera streams
- person detection and per-camera multi-object tracking on video
- inward-direction filtering
- known-face matching when usable
- `Unknown_Global_ID` creation and reuse across cameras
- map-aware travel-time constraints and cross-camera association
- event logs, snapshots, and identity timelines

## Current Status

The repository is no longer centered on the earlier GT-backed or single-source-only baseline.

The current active debug and benchmark phase is the **Wildtrack 4-camera ROI benchmark**:

1. use four different recorded Wildtrack videos: `C3`, `C5`, `C6`, `C7`
2. run real `YOLOv8n + ByteTrack` inference per camera
3. link short broken tracklets after ByteTrack
4. apply calibrated polygon ROI masking before ReID and association
5. keep only inward-direction events
6. run pose-aware face best-shot selection
7. fall back to OSNet body ReID when face is missing or rejected
8. hard-reject impossible topology/travel-time candidates before similarity scoring
9. export resolved events, best-shots, timelines, handoff summaries, and audit logs
10. benchmark the phase with a paired no-ROI versus ROI comparison

The earlier single-source sequential replay proof is still in the repo as a controlled sanity path, but it is no longer the main benchmark of the current phase.

## What Is Implemented

- real detector+tracker inference in the current runnable benchmark path
- short-gap tracklet linking after ByteTrack
- pose-aware face quality gate before best-shot selection
- pending association with timeout cleanup and UI-ready pending state
- OSNet body ReID as a real fallback path
- topology/travel-time hard filter before similarity
- detector/tracker cache for faster debug loops
- quantitative evaluation scripts and threshold-analysis tooling
- simulated real-time live file pipeline with latency tracing
- Unknown-ID timeline export and dashboard view
- ROI calibration tool served from the same lightweight demo server

## Current Bottlenecks

The project is improving, but it is not at a “done” state yet.

- ROI masking cuts false positives strongly, but the current official 4-camera benchmark still has negative `MOTA`
- the local detect+track stage is still the largest accuracy bottleneck
- the laptop remains CPU-bound for repeated inference and evaluation runs
- this repository is a thesis prototype, not a production CCTV system

The current short official 4-camera benchmark numbers are:

- before ROI: `FP=1805`, `FN=249`, `MOTA=-6.3238`
- after ROI: `FP=526`, `FN=250`, `MOTA=-1.977`

That means ROI masking is clearly helping, but it has **not yet** pushed the benchmark to positive `MOTA`.

## Key Runtime Pieces

- `insightface_demo_assets/runtime/run_offline_multicam_pipeline.py`
  - offline end-to-end orchestrator
- `insightface_demo_assets/runtime/offline_pipeline/`
  - offline event building and orchestration
- `insightface_demo_assets/runtime/association_core/`
  - topology, face/body evidence, pending, gallery lifecycle, and policy logic
- `insightface_demo_assets/runtime/run_face_resolution_demo.py`
  - identity-resolution compatibility entrypoint
- `insightface_demo_assets/runtime/live_pipeline/`
  - simulated real-time live/file ingestion path
- `insightface_demo_assets/runtime/run_live_event_demo_server.py`
  - lightweight UI server for latest events, timeline, and calibration tool

## Current Important Configs

Current 4-camera ROI benchmark:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.wildtrack_4cam_inference_roi_benchmark.yaml`
- `insightface_demo_assets/runtime/config/manual_scene_calibration.wildtrack_4cam_phase.yaml`
- `insightface_demo_assets/runtime/config/camera_transition_map.wildtrack_4cam_phase.yaml`

Paired 4-camera no-ROI baseline:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.wildtrack_4cam_inference_no_roi_benchmark.yaml`

Quantitative evaluation configs:

- `insightface_demo_assets/runtime/config/quantitative_evaluation.wildtrack_4cam_inference_roi_benchmark.yaml`
- `insightface_demo_assets/runtime/config/quantitative_evaluation.wildtrack_4cam_inference_no_roi_benchmark.yaml`

Earlier sequential proof-phase configs remain available:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_50s.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_90s.yaml`

## Runnable Commands

Default current benchmark:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_multicam_identity_demo.ps1"
```

Explicit 4-camera ROI benchmark:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_wildtrack_4cam_roi_benchmark.ps1"
```

Direct Python invocation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.wildtrack_4cam_inference_roi_benchmark.yaml"
```

Paired no-ROI baseline:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.wildtrack_4cam_inference_no_roi_benchmark.yaml"
```

Quantitative evaluation for the no-ROI baseline:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_quantitative_evaluation.py" --config ".\insightface_demo_assets\runtime\config\quantitative_evaluation.wildtrack_4cam_inference_no_roi_benchmark.yaml"
```

Quantitative evaluation for the ROI benchmark:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_quantitative_evaluation.py" --config ".\insightface_demo_assets\runtime\config\quantitative_evaluation.wildtrack_4cam_inference_roi_benchmark.yaml"
```

Calibration tool and timeline server:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_live_event_demo_server.ps1"
```

Earlier sequential replay proof-phase command:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_single_source_sequential_video_phase.ps1"
```

## Output Layout

Offline runs export to:

- `outputs/offline_runs/<run_name>/tracks/`
- `outputs/offline_runs/<run_name>/events/`
- `outputs/offline_runs/<run_name>/timelines/`
- `outputs/offline_runs/<run_name>/summaries/`
- `outputs/offline_runs/<run_name>/audit/`
- `outputs/offline_runs/<run_name>/association_logs/`
- `outputs/offline_runs/<run_name>/evaluation/`

Important artifacts in the current phase:

- `events/resolved_events.csv`
- `events/latest_events.json`
- `timelines/unknown_identity_timeline.json`
- `summaries/cross_camera_handoff_summary.json`
- `summaries/face_resolution_summary.json`
- `summaries/offline_pipeline_summary.json`
- `evaluation/quantitative_metrics_summary.json`

## Documentation

Core docs:

- [docs/offline_pipeline.md](docs/offline_pipeline.md)
- [docs/live_pipeline.md](docs/live_pipeline.md)
- [docs/live_demo_ui.md](docs/live_demo_ui.md)
- [docs/manual_scene_calibration.md](docs/manual_scene_calibration.md)
- [docs/quantitative_evaluation.md](docs/quantitative_evaluation.md)

Association-specific docs:

- [docs/association_paper_grounded_design.md](docs/association_paper_grounded_design.md)
- [docs/association_runtime_config.md](docs/association_runtime_config.md)
- [docs/association_trace_logging.md](docs/association_trace_logging.md)
- [docs/camera_transition_map_config.md](docs/camera_transition_map_config.md)
- [docs/camera_subzone_config.md](docs/camera_subzone_config.md)
- [docs/line_aware_best_shot.md](docs/line_aware_best_shot.md)
- [docs/phase_f_quality_diagnostic.md](docs/phase_f_quality_diagnostic.md)
- [docs/phase_f_c5_c6_map_fix.md](docs/phase_f_c5_c6_map_fix.md)
- [docs/phase_f_body_fallback.md](docs/phase_f_body_fallback.md)
- [docs/phase_f_tuning_rerun.md](docs/phase_f_tuning_rerun.md)
- [docs/offline_multiprocessing_architecture.md](docs/offline_multiprocessing_architecture.md)
- [docs/association_evaluation_tuning.md](docs/association_evaluation_tuning.md)
- [docs/single_source_sequential_sanity_demo.md](docs/single_source_sequential_sanity_demo.md)
