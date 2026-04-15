# A Multi-Camera System for Detecting and Tracking Strangers Entering a Facility

Graduation project repository for a multi-camera security pipeline that:

- processes 4 camera streams
- keeps only people moving into a protected area
- detects and tracks people per camera
- matches faces against a known database using pretrained models
- creates stranger / unknown identities when no known match exists
- reuses `Unknown_Global_ID` across cameras using map-aware cross-camera association
- stores event logs, snapshots, and identity timelines

## Current Status

The current default runnable vertical slice is the supervisor-approved Cam6 video proof phase:

1. take one source video from `C6`
2. replay it sequentially 4 times as virtual cameras `C1 -> C2 -> C3 -> C4`
3. inject fake travel-time offsets between the 4 passes
4. process a verified `~50` second actual-video window from the source as the current proof window
4. keep only inward-direction entry events
5. run face-first known matching and unknown creation
6. reuse `Unknown_Global_ID` across the 4 virtual passes through the current association core
7. run short-gap tracklet linking after ByteTrack so fragmented intra-camera tracklets are cleaned before association
8. reject face frames with bad pose before they enter the best-shot buffer
9. keep ambiguous cross-camera candidates in `PENDING` until they resolve or time out cleanly
10. export resolved events, timelines, mapping tables, and audit logs into a single run folder

This phase is intentionally simpler than the harder real multi-camera case. The goal is to prove that the core reuse logic works in the easiest sequential sanity scenario before returning to more difficult multi-camera conditions.

Important limitation:

- the default offline replay path now uses real detector+tracker inference, but it is still CPU-bound on this laptop
- the replay path now adds short-gap intra-camera tracklet linking immediately after ByteTrack to reduce downstream ID noise
- face best-shot buffering is now pose-aware: sharp frames with `|yaw| > 30°` or `|pitch| > 20°` are rejected before embedding / gallery storage
- pending association entries are now garbage-collected after `2s` if they never gather enough evidence, and the live demo keeps them in an `Analyzing...` state instead of showing a final ID too early
- a longer `90` second actual-video config exists for the same replay mode, but it exceeded the current local timeout budget during verification

## Association Source of Truth

Association design must follow:

- [docs/association_paper_grounded_design.md](docs/association_paper_grounded_design.md)

Association must not fall back to a vague weighted sum if the design note already defines the logic. The required structure is:

1. `quality gate`
2. `candidate filtering` by topology + travel time + zone constraints when available, with subzone constraints when the dataset provides them
3. `modality-aware appearance evidence` with face and body kept separate
4. `accept / reject / create / defer`
5. `gallery lifecycle` with TTL and top-k references

The association layer must support:

- `overlap`
- `sequential`
- `weak_link`

and it must emit decision logs with `reason_code`.

Additional docs:

- [docs/association_runtime_config.md](docs/association_runtime_config.md)
- [docs/association_trace_logging.md](docs/association_trace_logging.md)
- [docs/camera_transition_map_config.md](docs/camera_transition_map_config.md)
- [docs/camera_subzone_config.md](docs/camera_subzone_config.md)
- [docs/line_aware_best_shot.md](docs/line_aware_best_shot.md)
- [docs/phase_f_quality_diagnostic.md](docs/phase_f_quality_diagnostic.md)
- [docs/phase_f_c5_c6_map_fix.md](docs/phase_f_c5_c6_map_fix.md)
- [docs/phase_f_body_fallback.md](docs/phase_f_body_fallback.md)
- [docs/phase_f_tuning_rerun.md](docs/phase_f_tuning_rerun.md)
- [docs/offline_pipeline.md](docs/offline_pipeline.md)
- [docs/single_source_sequential_sanity_demo.md](docs/single_source_sequential_sanity_demo.md)
- [docs/offline_multiprocessing_architecture.md](docs/offline_multiprocessing_architecture.md)
- [docs/association_evaluation_tuning.md](docs/association_evaluation_tuning.md)
- [docs/live_pipeline.md](docs/live_pipeline.md)
- [docs/live_demo_ui.md](docs/live_demo_ui.md)
- [docs/manual_scene_calibration.md](docs/manual_scene_calibration.md)

## Phase Status

Current code status:

- `insightface_demo_assets/runtime/run_face_resolution_demo.py` remains the face-resolution compatibility entrypoint
- `insightface_demo_assets/runtime/run_offline_multicam_pipeline.py` is the offline end-to-end orchestrator entrypoint
- `insightface_demo_assets/runtime/offline_pipeline/` owns offline event-building and orchestration
- the current default offline demo backend is `single_source_sequential_replay`
- the old Wildtrack 4-source flow remains available as a reference backend, but it is no longer the default defense/demo command
- the offline orchestrator now supports both `sequential` and `multiprocessing` execution modes
- manual scene calibration is now the required runtime ROI path through `insightface_demo_assets/runtime/config/manual_scene_calibration.wildtrack.json`
- deprecated auto/inferred ROI fallback is disabled in the runtime path; missing calibration now fails clearly instead of silently using old defaults
- association core logic lives under `insightface_demo_assets/runtime/association_core/`
- `insightface_demo_assets/runtime/run_association_tuning.py` now provides a cache-first threshold tuning workflow
- association thresholds and policies are externalized via `insightface_demo_assets/runtime/config/association_policy.example.yaml`
- the current selected policy for Wildtrack demo runs is `insightface_demo_assets/runtime/config/association_policy.wildtrack_tuned.yaml`
- the current selected policy after Phase F is `insightface_demo_assets/runtime/config/association_policy.wildtrack_phase_f_tuned.yaml`
- camera-pair transitions, zones, and subzones are externalized via `insightface_demo_assets/runtime/config/camera_transition_map.example.yaml`
- entry-camera best-shot selection is now line-aware and subzone-aware through `wildtrack_demo/wildtrack_demo_config.json`
- direction filtering now uses trajectory/momentum history plus line and zone context instead of one-frame line crossing only
- ROI masks are now applied in the live path to reduce wasted detection work outside the calibrated processing polygon
- the current verified inference video-phase config now lives under `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_50s.yaml`
- a longer unverified stress config also lives under `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_90s.yaml`
- the earlier short sanity config still lives under `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6.yaml`
- a reference Wildtrack multi-source config still lives under `insightface_demo_assets/runtime/config/offline_pipeline_demo.example.yaml`
- association decision logs are exported under `association_logs/` in each offline run
- event-generation audit now records best-shot strategy, subzone choice, and frames after anchor
- body-first fallback is now part of the default association flow when face is missing or unreliable
- offline runs export standardized folders under `outputs/offline_runs/`
- scenario tests for association core and offline event creation live under `tests/`
- live ingestion baseline now lives under `insightface_demo_assets/runtime/live_pipeline/`
- live outputs are exported under `outputs/live_runs/`
- lightweight web demo now reads the live event outputs through a small local HTTP server
- the same lightweight web demo server now also exposes a manual calibration page at `/calibration.html`

## Project Scope

In scope:

- 4 camera streams
- inward-direction filtering
- per-camera detection and tracking
- known face matching using pretrained models
- unknown identity creation and reuse across cameras
- shared camera map and expected travel time
- event log, snapshots, and appearance timeline

Out of scope for now:

- training large face recognition models from scratch
- heavy end-to-end retraining on the current laptop
- live RTSP-first deployment before a stable video-file vertical slice exists

## Current Runnable Commands

From `cmd`:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_multicam_identity_demo.ps1"
```

Default Cam6 replay video-phase demo:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_single_source_sequential_video_phase.ps1"
```

That command runs one physical source video, replays it sequentially as virtual cameras `C1`, `C2`, `C3`, `C4`, and injects fake travel-time offsets from the verified inference `~50` second config.

Direct offline orchestrator:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_offline_multicam_pipeline.ps1"
```

Direct Python invocation for the same video-phase config:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.single_source_sequential_c6_inference_50s.yaml"
```

Low-load sanity run:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.low_load.yaml"
```

Live file-sanity run:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_live_multicam_demo.ps1"
```

Direct live orchestrator:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_live_multicam_demo.py" --config ".\insightface_demo_assets\runtime\config\live_pipeline_demo.file_sanity.yaml"
```

Start the lightweight web demo:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_live_event_demo_server.ps1"
```

Low-load multiprocessing sanity run:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.multiprocessing.low_load.yaml"
```

Run association tuning from cached candidate events:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_association_tuning.py" --config ".\insightface_demo_assets\runtime\config\association_tuning_grid.example.yaml"
```

To run only the face-resolution stage:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_face_resolution_demo.py"
```

Main outputs:

- `outputs/offline_runs/<run_name>/events/`
- `outputs/offline_runs/<run_name>/timelines/`
- `outputs/offline_runs/<run_name>/summaries/`
- `outputs/offline_runs/<run_name>/audit/`
- `outputs/offline_runs/<run_name>/association_logs/`
- `outputs/offline_runs/<run_name>/events/unknown_id_mapping.csv`
- `outputs/offline_runs/<run_name>/summaries/face_body_usage_summary.json`
- `outputs/live_runs/<run_name>/events/`
- `outputs/live_runs/<run_name>/association_logs/`
- `outputs/live_runs/<run_name>/summaries/`
- `insightface_demo_assets/runtime/face_resolution_summary.json`
- `insightface_demo_assets/runtime/stream_identity_timeline.csv`
- `insightface_demo_assets/runtime/audit_report.md`

Run the lightweight tests:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" -m pytest tests\test_association_core.py tests\test_offline_pipeline.py
```

## Current Repository Layout

```text
.
|- README.md
|- run_multicam_identity_demo.ps1
|- run_offline_multicam_pipeline.ps1
|- docs/
|  |- association_paper_grounded_design.md
|  `- offline_pipeline.md
|- wildtrack_demo/
|  |- wildtrack_demo_config.json
|  |- export_wildtrack_demo.ps1
|  `- output/
|- insightface_demo_assets/
|  |- known_faces/
|  |- known_face_manifest.csv
|  |- runtime/
|     |- live_pipeline/
|     |- offline_pipeline/
|     |- association_core/
|     |- config/
|     |  |- association_policy.example.yaml
|     |  |- association_policy.wildtrack_tuned.yaml
|     |  |- association_tuning_grid.example.yaml
|     |  |- camera_transition_map.example.yaml
|     |  |- live_pipeline_demo.file_sanity.yaml
|     |  |- live_pipeline_demo.rtsp.example.yaml
|     |  |- offline_pipeline_demo.example.yaml
|     |  |- offline_pipeline_demo.low_load.yaml
|     |  `- offline_pipeline_demo.multiprocessing.low_load.yaml
|     |- face_demo_config.json
|     |- run_association_tuning.py
|     |- run_face_resolution_demo.py
|     |- run_live_event_demo_server.py
|     |- run_live_multicam_demo.py
|     `- run_offline_multicam_pipeline.py
|- outputs/
|  `- offline_runs/
|  `- live_runs/
|- tests/
|  |- conftest.py
|  |- test_association_core.py
|  |- test_live_demo_server.py
|  |- test_live_pipeline.py
|  `- test_offline_pipeline.py
|- requirements-dev.txt
|- Wildtrack/
|- PAPERS/
|- stranger_demo_bootstrap/
`- Dataset/
```

Notes:

- `run_multicam_identity_demo.ps1` now routes to the offline orchestrator so the legacy command still works.
- `run_multicam_identity_demo.ps1` and `run_offline_multicam_pipeline.ps1` now resolve paths relative to the script location instead of relying on hard-coded absolute Windows paths.
- `run_face_resolution_demo.py` remains the stage-only entrypoint for face resolution and association.
- `runtime/config/association_policy.example.yaml` is the public policy template for thresholds, TTL, margins, and defer/create rules.
- `runtime/config/association_policy.wildtrack_tuned.yaml` is the pre-Phase-F selected policy kept for reference.
- `runtime/config/association_policy.wildtrack_phase_f_tuned.yaml` is the current selected policy after the Phase F body-fallback and map-fix rerun.
- `runtime/config/camera_transition_map.example.yaml` is the public map-aware template for camera-pair transitions, entry/exit zones, overlap behavior, and subzones.
- `wildtrack_demo/wildtrack_demo_config.json` now carries line-aware best-shot preferences such as preferred subzone types and minimum frames after an anchor crossing.
- `runtime/config/association_tuning_grid.phase_f.yaml` is the Phase F tuning sweep config for regenerated candidate events.
- `runtime/config/offline_pipeline_demo.example.yaml` is the public offline run template.
- `runtime/config/offline_pipeline_demo.multiprocessing.low_load.yaml` is the lightweight producer-consumer sample config.
- `runtime/config/live_pipeline_demo.file_sanity.yaml` is the current live-ingestion sanity config.
- `runtime/config/live_pipeline_demo.rtsp.example.yaml` and `runtime/config/live_pipeline_demo.webcam.example.yaml` are live source templates.
- `runtime/live_pipeline/orchestrator.py` is the lightweight live producer-consumer baseline.
- `runtime/run_live_event_demo_server.py` serves a minimal live event UI from the generated outputs.
- `runtime/config/association_tuning_grid.example.yaml` is the public tuning sweep template.
- `runtime/association_logs/` and `outputs/offline_runs/` are generated at run time and are intentionally not part of source control.
- `outputs/live_runs/` is generated at run time and is intentionally not part of source control.
- `stranger_demo_bootstrap/` remains the scaffold for the later structured repo split.
- `Dataset/` is legacy/reference material and is not the current runnable thesis path.

## Planned Incremental Phases

Near-term implementation should proceed in small runnable phases:

1. keep the current offline video-file vertical slice runnable
2. add multiprocessing producer-consumer around the offline flow
3. add evaluation and threshold tuning from cached offline outputs
4. improve best-shot selection without adding heavy new models
5. extend the current live baseline to cleaner real-camera demos without changing the association core

## Dependencies

Current demo runtime depends on:

- Python venv at `.venv_insightface_demo`
- InsightFace model cache at `%USERPROFILE%\.insightface\models\buffalo_l` on Windows, or `~/.insightface/models/buffalo_l` on Unix-like systems
- Wildtrack demo assets already exported in this repo

If new dependencies or configs are added in future phases, they must be documented and committed together with sample config files.
