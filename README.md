# A Multi-Camera System for Detecting and Tracking Strangers Entering a Facility

Graduation project repository for a multi-camera security pipeline that stays within the original scope:

- 4 camera streams
- person detection and per-camera multi-object tracking on video
- inward-direction filtering
- known-face matching when usable
- `Unknown_Global_ID` creation and reuse across cameras
- map-aware travel-time constraints and cross-camera association
- event logs, snapshots, and identity timelines

## Current Active Dataset

The active dataset is now the **self-recorded New Dataset**, not Wildtrack.

Current physical capture status:

- `New Dataset/Camera 1`
- `New Dataset/Camera 2`

Each physical camera folder stores paired clips with the same stem:

- `Camera 1/a1.mp4`
- `Camera 2/a1.mp4`
- later pairs should follow the same rule: `a2`, `b1`, ...

Meaning of a pair:

- `Camera 1/<stem>`: stranger approaching / entering the facility door
- `Camera 2/<stem>`: the same stranger appearing inside after crossing the door

The thesis scope still stays at 4-camera demonstration level. Because the current self-recorded dataset only has **2 physical cameras**, the repository now uses a **logical 4-camera demo expansion**:

- `C1` -> physical Camera 1
- `C2` -> physical Camera 2
- `C3` -> logical delayed replay of physical Camera 1
- `C4` -> logical delayed replay of physical Camera 2

This is documented explicitly as a demo adapter for scope alignment. It is **not** presented as 4 independent physical cameras.

## What Stays Unchanged

The phase that migrates away from Wildtrack does **not** rewrite the core pipeline.

The runtime order is still:

`Detect -> Track -> Filter IN direction -> Match Face -> Manage Unknown ID -> Cross-camera Association`

The core logic remains configuration-driven and separate from dataset profiles:

- IoU
- cosine similarity
- travel-time logic
- point-in-polygon
- direction logic
- association decision logic

## What Changed In The Current Phase

This phase does not add a new product surface. It closes the algorithmic gaps that were
still masking weak association quality:

- `sequential.body_primary` is restored to `0.72`; the repo no longer relies on the earlier
  `0.55` shortcut
- body ReID for cross-camera reuse is now **tracklet-based**, not single-frame-based
- body crops are filtered and normalized before ReID embedding extraction
- high-angle cameras (`C1`, `C3`) are now treated as **event/body anchors**, not face sources
- eye-level cameras (`C2`, `C4`) remain face candidates, but only through a strict
  best-shot gate with bbox-size and yaw/pitch limits
- topology / travel-time / zone / subzone are now an explicit decision signal for
  near-threshold sequential reuse, instead of being only a silent hard filter
- direction validation and body-tracklet comparison now have standalone scripts and artifacts

## Current Important Configs

Active New Dataset configs:

- `insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/camera_transition_map.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/bytetrack.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml`

Legacy Wildtrack configs remain in the repo as reference/regression assets, but they are no longer the active default narrative.

The most important policy values in the current phase are:

- `association_policy.new_dataset_demo.yaml`
  - `decision_policy.relation_thresholds.sequential.body_primary = 0.72`
  - `decision_policy.topology_supported_accept.enabled = true`
  - `quality_gate.min_face_bbox_width / height / area`
  - `body_reid.tracklet_pooling_*`
- `dataset_profile.new_dataset_demo.yaml`
  - `face_capture_mode`
  - `anchor_point_mode`

## Default Commands

Default current offline demo:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_multicam_identity_demo.ps1"
```

Explicit New Dataset logical 4-camera demo:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_new_dataset_logical_demo.ps1"
```

Direct Python invocation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml"
```

Independent direction validation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_direction_validation.py" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --output-root "outputs/evaluations/direction_validation_tracklet_phase"
```

Body tracklet comparison against an offline run:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_body_tracklet_evaluation.py" --run-output-root "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4" --output-dir "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet"
```

Timeline + calibration server against the current New Dataset output root:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_live_event_demo_server.ps1"
```

Regression tests:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" -m pytest tests -q
```

## Current Validation Snapshot

The current phase has been validated on the available self-recorded pair `a1`:

- direction validation: `5 / 5` cases passed
- body tracklet evaluation:
  - average old single-frame body score: `0.6105`
  - average new tracklet body score: `0.6298`
- full offline smoke (`new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4`)
  - `TOTAL_EVENTS = 3`
  - `UNIQUE_UNKNOWN_IDS = 1`
  - `UNKNOWN_REUSE_COUNT = 2`
  - `handoff_edge_count = 2`
  - identity sequence reused across `C1 -> C2 -> C3`

Important interpretation:

- reuse now succeeds without lowering `body_primary`
- the deciding path is **strong topology/time/zone/subzone support + tracklet-based body evidence**
- face still contributes audit and rejection reasons, but it does not yet create usable
  embeddings on the current `a1` smoke

## Output Artifacts

Current phase artifacts are written under:

- `outputs/evaluations/direction_validation_tracklet_phase/`
- `outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/`
- `outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet/`

The most useful files are:

- `summaries/face_resolution_summary.json`
- `summaries/face_body_usage_summary.json`
- `summaries/cross_camera_handoff_summary.json`
- `runtime/association_logs/association_decisions.jsonl`
- `evaluation/body_tracklet/body_tracklet_comparison_summary.json`

## Current Constraints

- the active self-recorded dataset currently has only 2 physical cameras
- clip coverage is still very small
- only `a1` is available locally right now; `a2` and `b1` are not present yet, so this phase
  cannot honestly report quantitative results for those clips
- logical `C3/C4` are explicit demo replays derived from `C1/C2`
- Wildtrack evaluation utilities and benchmark assets still remain in the repo for legacy comparison
- face embeddings are still absent on the current `a1` smoke because the usable face shots are
  rejected by camera-role policy or yaw limits
- the current topology-supported accept path is intended for strong sequential cases with
  near-threshold body evidence; it is not a license to weaken map constraints or lower
  thresholds globally
- this repository is a thesis prototype, not a production CCTV system

## Documentation

- [docs/offline_pipeline.md](docs/offline_pipeline.md)
- [docs/new_dataset_demo.md](docs/new_dataset_demo.md)
- [docs/new_dataset_algorithmic_audit_phase_tracklet_face_topology.md](docs/new_dataset_algorithmic_audit_phase_tracklet_face_topology.md)
- [docs/live_demo_ui.md](docs/live_demo_ui.md)
- [docs/manual_scene_calibration.md](docs/manual_scene_calibration.md)
- [docs/association_runtime_config.md](docs/association_runtime_config.md)
- [docs/association_trace_logging.md](docs/association_trace_logging.md)
- [docs/quantitative_evaluation.md](docs/quantitative_evaluation.md)
