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

The current runnable vertical slice is video-file-first:

1. run one offline orchestrator over 4 selected Wildtrack video sources
2. generate GT-backed per-camera track rows and entry events
3. run face matching with InsightFace and a known gallery
4. run paper-grounded cross-camera association through a compatibility entrypoint
5. create or reuse unknown IDs
6. export resolved events, timelines, and audit logs into a single run folder

RTSP/live ingestion is planned later. Video files are the current priority.

Important limitation:

- the current offline detect/track stage is still a Wildtrack annotation-backed provider for the thesis demo baseline
- the repo now has a real offline end-to-end flow, but it is not yet a production detector+tracker inference stack

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
- [docs/offline_pipeline.md](docs/offline_pipeline.md)
- [docs/offline_multiprocessing_architecture.md](docs/offline_multiprocessing_architecture.md)
- [docs/association_evaluation_tuning.md](docs/association_evaluation_tuning.md)

## Phase Status

Current code status:

- `insightface_demo_assets/runtime/run_face_resolution_demo.py` remains the face-resolution compatibility entrypoint
- `insightface_demo_assets/runtime/run_offline_multicam_pipeline.py` is the offline end-to-end orchestrator entrypoint
- `insightface_demo_assets/runtime/offline_pipeline/` owns offline event-building and orchestration
- the offline orchestrator now supports both `sequential` and `multiprocessing` execution modes
- association core logic lives under `insightface_demo_assets/runtime/association_core/`
- `insightface_demo_assets/runtime/run_association_tuning.py` now provides a cache-first threshold tuning workflow
- association thresholds and policies are externalized via `insightface_demo_assets/runtime/config/association_policy.example.yaml`
- the current selected policy for Wildtrack demo runs is `insightface_demo_assets/runtime/config/association_policy.wildtrack_tuned.yaml`
- camera-pair transitions, zones, and subzones are externalized via `insightface_demo_assets/runtime/config/camera_transition_map.example.yaml`
- offline run config now lives under `insightface_demo_assets/runtime/config/offline_pipeline_demo.example.yaml`
- association decision logs are exported under `association_logs/` in each offline run
- offline runs export standardized folders under `outputs/offline_runs/`
- scenario tests for association core and offline event creation live under `tests/`
- live ingestion, dashboard, and storage remain later phases

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
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
powershell -ExecutionPolicy Bypass -File ".\run_multicam_identity_demo.ps1"
```

Direct offline orchestrator:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
powershell -ExecutionPolicy Bypass -File ".\run_offline_multicam_pipeline.ps1"
```

Low-load sanity run:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.low_load.yaml"
```

Low-load multiprocessing sanity run:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.multiprocessing.low_load.yaml"
```

Run association tuning from cached candidate events:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_association_tuning.py" --config ".\insightface_demo_assets\runtime\config\association_tuning_grid.example.yaml"
```

To run only the face-resolution stage:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_face_resolution_demo.py"
```

Main outputs:

- `outputs/offline_runs/<run_name>/events/`
- `outputs/offline_runs/<run_name>/timelines/`
- `outputs/offline_runs/<run_name>/summaries/`
- `outputs/offline_runs/<run_name>/audit/`
- `outputs/offline_runs/<run_name>/association_logs/`
- `insightface_demo_assets/runtime/face_resolution_summary.json`
- `insightface_demo_assets/runtime/stream_identity_timeline.csv`
- `insightface_demo_assets/runtime/audit_report.md`

Run the lightweight tests:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
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
|     |- offline_pipeline/
|     |- association_core/
|     |- config/
|     |  |- association_policy.example.yaml
|     |  |- association_policy.wildtrack_tuned.yaml
|     |  |- association_tuning_grid.example.yaml
|     |  |- camera_transition_map.example.yaml
|     |  |- offline_pipeline_demo.example.yaml
|     |  |- offline_pipeline_demo.low_load.yaml
|     |  `- offline_pipeline_demo.multiprocessing.low_load.yaml
|     |- face_demo_config.json
|     |- run_association_tuning.py
|     |- run_face_resolution_demo.py
|     `- run_offline_multicam_pipeline.py
|- outputs/
|  `- offline_runs/
|- tests/
|  |- conftest.py
|  |- test_association_core.py
|  `- test_offline_pipeline.py
|- requirements-dev.txt
|- Wildtrack/
|- PAPERS/
|- stranger_demo_bootstrap/
`- Dataset/
```

Notes:

- `run_multicam_identity_demo.ps1` now routes to the offline orchestrator so the legacy command still works.
- `run_face_resolution_demo.py` remains the stage-only entrypoint for face resolution and association.
- `runtime/config/association_policy.example.yaml` is the public policy template for thresholds, TTL, margins, and defer/create rules.
- `runtime/config/association_policy.wildtrack_tuned.yaml` is the current selected policy from cached offline tuning experiments.
- `runtime/config/camera_transition_map.example.yaml` is the public map-aware template for camera-pair transitions, entry/exit zones, overlap behavior, and subzones.
- `runtime/config/offline_pipeline_demo.example.yaml` is the public offline run template.
- `runtime/config/offline_pipeline_demo.multiprocessing.low_load.yaml` is the lightweight producer-consumer sample config.
- `runtime/config/association_tuning_grid.example.yaml` is the public tuning sweep template.
- `runtime/association_logs/` and `outputs/offline_runs/` are generated at run time and are intentionally not part of source control.
- `stranger_demo_bootstrap/` remains the scaffold for the later structured repo split.
- `Dataset/` is legacy/reference material and is not the current runnable thesis path.

## Planned Incremental Phases

Near-term implementation should proceed in small runnable phases:

1. keep the current offline video-file vertical slice runnable
2. add multiprocessing producer-consumer around the offline flow
3. add evaluation and threshold tuning from cached offline outputs
4. improve best-shot selection without adding heavy new models
5. add RTSP/live only after the file-based pipeline is stable

## Dependencies

Current demo runtime depends on:

- Python venv at `.venv_insightface_demo`
- InsightFace model cache at `C:\Users\Admin\.insightface\models\buffalo_l`
- Wildtrack demo assets already exported in this repo

If new dependencies or configs are added in future phases, they must be documented and committed together with sample config files.
