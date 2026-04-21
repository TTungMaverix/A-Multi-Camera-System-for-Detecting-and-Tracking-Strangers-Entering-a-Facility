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

## What Changed In This Dataset-Migration Phase

- Wildtrack is no longer assumed as the only dataset profile
- dataset-specific paths and camera assumptions are moved into profile/config files
- the new dataset has a stem-based pair adapter instead of hardcoded `a1`
- per-camera ROI anchor mode is now configurable
  - `bottom_center` for Camera 1 style views
  - `center_center` for Camera 2 style partial-body views
- topology is rewritten around **travel time + reachability**, not overlap
- the default offline demo path now targets the New Dataset logical 4-camera adapter
- README and offline pipeline docs now describe the new dataset as the active dataset

## Current Important Configs

Active New Dataset configs:

- `insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/camera_transition_map.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/bytetrack.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml`

Legacy Wildtrack configs remain in the repo as reference/regression assets, but they are no longer the active default narrative.

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

Timeline + calibration server against the current New Dataset output root:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_live_event_demo_server.ps1"
```

## Current Constraints

- the active self-recorded dataset currently has only 2 physical cameras
- clip coverage is still small, so this phase is about dataset migration and demo preparation, not a full benchmark rerun
- logical `C3/C4` are explicit demo replays derived from `C1/C2`
- Wildtrack evaluation utilities and benchmark assets still remain in the repo for legacy comparison
- this repository is a thesis prototype, not a production CCTV system

## Documentation

- [docs/offline_pipeline.md](docs/offline_pipeline.md)
- [docs/new_dataset_demo.md](docs/new_dataset_demo.md)
- [docs/live_demo_ui.md](docs/live_demo_ui.md)
- [docs/manual_scene_calibration.md](docs/manual_scene_calibration.md)
- [docs/quantitative_evaluation.md](docs/quantitative_evaluation.md)
