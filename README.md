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

1. generate candidate entry and follow-up events from 4 selected Wildtrack cameras
2. run face matching with InsightFace and a known gallery
3. run paper-grounded cross-camera association through a compatibility entrypoint
4. create or reuse unknown IDs
5. export resolved events, timelines, and audit logs

RTSP/live ingestion is planned later. Video files are the current priority.

## Association Source of Truth

Association design must follow:

- [docs/association_paper_grounded_design.md](docs/association_paper_grounded_design.md)

Association must not fall back to a vague weighted sum if the design note already defines the logic. The required structure is:

1. `quality gate`
2. `candidate filtering` by topology + travel time + optional zone constraints
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

## Phase Status

Current code status:

- `insightface_demo_assets/runtime/run_face_resolution_demo.py` remains the runnable compatibility entrypoint
- association core logic now lives under `insightface_demo_assets/runtime/association_core/`
- association thresholds and policies are externalized via `insightface_demo_assets/runtime/config/association_policy.example.yaml`
- association decision logs are exported under `insightface_demo_assets/runtime/association_logs/`
- scenario tests for association core live under `tests/`
- this phase refactors only the association core
- ingest, detector, tracker, dashboard, and the demo command are intentionally kept stable in this phase

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

To run only the face-resolution stage:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_face_resolution_demo.py"
```

Main outputs:

- `wildtrack_demo/output/events/`
- `insightface_demo_assets/runtime/face_resolution_summary.json`
- `insightface_demo_assets/runtime/stream_identity_timeline.csv`
- `insightface_demo_assets/runtime/audit_report.md`
- `insightface_demo_assets/runtime/association_logs/association_decisions.jsonl`
- `insightface_demo_assets/runtime/association_logs/association_summary.json`

Run the lightweight association tests:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" -m pytest tests\test_association_core.py
```

## Current Repository Layout

```text
.
|- README.md
|- run_multicam_identity_demo.ps1
|- docs/
|  |- association_paper_grounded_design.md
|- wildtrack_demo/
|  |- wildtrack_demo_config.json
|  |- export_wildtrack_demo.ps1
|  |- output/
|- insightface_demo_assets/
|  |- known_faces/
|  |- known_face_manifest.csv
|  |- runtime/
|     |- association_core/
|     |  |- quality_gate.py
|     |  |- topology_filter.py
|     |  |- appearance_evidence.py
|     |  |- gallery_lifecycle.py
|     |  |- decision_policy.py
|     |  |- config_loader.py
|     |  `- trace_logging.py
|     |- config/
|     |  `- association_policy.example.yaml
|     |- face_demo_config.json
|     |- run_face_resolution_demo.py
|     |- audit_*.csv/json/md
|     `- resolved_events_mode_*.csv
|- tests/
|  |- conftest.py
|  `- test_association_core.py
|- requirements-dev.txt
|- Wildtrack/
|- PAPERS/
|- stranger_demo_bootstrap/
`- Dataset/
```

Notes:

- `run_face_resolution_demo.py` is still the public entrypoint for the demo phase.
- `association_core/` is the new paper-grounded association package used under that entrypoint.
- `runtime/config/association_policy.example.yaml` is the public policy template for thresholds, TTL, margins, and defer/create rules.
- `runtime/association_logs/` is generated at run time and is intentionally not part of source control.
- `stranger_demo_bootstrap/` remains the scaffold for the later structured repo split.
- `Dataset/` is legacy/reference material and is not the current runnable thesis path.

## Planned Incremental Phases

Near-term implementation should proceed in small runnable phases:

1. keep the current video-file vertical slice runnable
2. refactor association core without touching the rest of the pipeline in the same phase
3. keep the compatibility entrypoint working while moving logic into:
   - `quality_gate`
   - `topology_filter`
   - `appearance_evidence`
   - `gallery_lifecycle`
   - `decision_policy`
4. externalize thresholds, margins, TTL, and defer policy into config files
5. keep README and docs synchronized after each phase
6. add RTSP/live only after the file-based pipeline is stable

## Dependencies

Current demo runtime depends on:

- Python venv at `.venv_insightface_demo`
- InsightFace model cache at `C:\Users\Admin\.insightface\models\buffalo_l`
- Wildtrack demo assets already exported in this repo

If new dependencies or configs are added in future phases, they must be documented and committed together with sample config files.
