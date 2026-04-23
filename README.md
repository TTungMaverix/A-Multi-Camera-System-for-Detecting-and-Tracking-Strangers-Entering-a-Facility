# A Multi-Camera System for Detecting and Tracking Strangers Entering a Facility

Graduation project repository for a multi-camera stranger-tracking pipeline that stays within the thesis scope:

- 4 camera streams at the system-design level
- person detection and per-camera multi-object tracking on video
- inward-direction filtering
- known-face matching when usable
- `Unknown_Global_ID` creation and reuse across cameras
- map-aware travel-time constraints and cross-camera association
- event logs, snapshots, and identity timelines

## Current Active Dataset

The active dataset is the self-recorded **New Dataset**, not Wildtrack.

Physical folders:

- `New Dataset/Camera 1`
- `New Dataset/Camera 2`

Current local paired clips:

- `a1`
- `a2`
- `a3`
- `b1`

The runtime pairs clips by shared stem across the two physical camera folders, for example:

- `Camera 1/a1.mp4` <-> `Camera 2/a1.mp4`
- `Camera 1/a2.mp4` <-> `Camera 2/a2.mp4`

Future clips should keep the same shared-stem rule, for example `a4`, `b2`, and so on.

The thesis scope still stays at 4-camera demonstration level. Because the current active dataset only has **2 physical cameras**, the repo uses an explicit logical 4-camera expansion:

- `C1` -> physical Camera 1
- `C2` -> physical Camera 2
- `C3` -> logical delayed replay of physical Camera 1
- `C4` -> logical delayed replay of physical Camera 2

This is a documented demo adapter. It is **not** presented as 4 independent physical cameras.

## Calibration Reuse Policy

Calibration is treated as **per source camera**, not per clip.

That means:

- all clips in `New Dataset/Camera 1` reuse the Camera 1 calibration geometry
- all clips in `New Dataset/Camera 2` reuse the Camera 2 calibration geometry

This is the correct policy when the camera source is unchanged:

- same camera position
- same camera angle
- same background geometry
- same normalized coordinate space

The current inventory audit confirms that all local clips `a1`, `a2`, `a3`, and `b1` can reuse the existing calibration. There is no need to redraw ROI/line/zone/subzone per clip unless the source-camera geometry changes materially.

## Current Pipeline

The runtime order stays unchanged:

`Detect -> Track -> Filter IN direction -> Match Face -> Manage Unknown ID -> Cross-camera Association`

The repo still keeps the same architectural principles:

- quality gate before evidence use
- topology / travel-time / zone / subzone candidate filtering
- modality-aware face/body evidence
- gallery lifecycle with TTL and top-k references
- explicit accept / reject / create / defer logic
- reason-coded association logs

## What This Phase Changed

This phase does not add a new product surface. It focuses on dataset-aware evaluation coverage and stronger appearance evaluation.

What changed:

- added a dataset inventory harness for the local self-recorded clips
- added a calibration-reuse audit so new clips can reuse source-camera calibration without manual duplication
- extended evaluation from a single clip to all currently paired local clips
- kept `sequential.body_primary = 0.72` with no threshold rollback
- upgraded body tracklet pooling from mean-only baseline to **quality-aware pooling**
- benchmarked a stronger body extractor variant side-by-side (`osnet_x1_0`) against the current default (`osnet_x0_25`)
- kept topology/time as a strong signal, but now reports appearance-only vs topology-supported outcomes explicitly
- kept face gate strict, while auditing whether face-friendly cameras actually yield usable best shots

## Current Important Configs

Active New Dataset configs:

- `insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/camera_transition_map.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/bytetrack.new_dataset_demo.yaml`
- `insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml`

Important current policy keys:

- `body_reid.extractor_name`
- `body_reid.tracklet_pooling_mode`
- `body_reid.tracklet_pooling_quality_weights`
- `decision_policy.relation_thresholds.sequential.body_primary`
- `decision_policy.topology_supported_accept`
- `quality_gate.min_face_bbox_width / height / area`
- `dataset_profile.cameras[].anchor_point_mode`
- `dataset_profile.cameras[].face_capture_mode`

## Default Commands

Dataset inventory + calibration reuse audit:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_inventory.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --output-dir "outputs/evaluations/new_dataset_inventory_phase_current"
```

Independent direction validation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_direction_validation.py" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --output-root "outputs/evaluations/direction_validation_tracklet_phase"
```

Per-clip evaluation across all currently paired local clips:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_evaluation.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --inventory-json ".\outputs\evaluations\new_dataset_inventory_phase_current\dataset_inventory.json" --calibration-reuse-json ".\outputs\evaluations\new_dataset_inventory_phase_current\calibration_reuse_summary.json" --output-dir ".\outputs\evaluations\new_dataset_quality_pooling_phase_current"
```

Body tracklet comparison against an existing run root:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_body_tracklet_evaluation.py" --run-output-root "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4" --output-dir "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet_phase_current"
```

Current offline smoke demo:

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_new_dataset_logical_demo.ps1"
```

Direct offline orchestrator run:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml"
```

Regression tests:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" -m pytest tests -q
```

## Current Validation Snapshot

Current local coverage from `outputs/evaluations/new_dataset_inventory_phase_current/dataset_inventory.json`:

- total paired clips available: `4`
- paired clips ready for evaluation: `4`
- missing paired clips to reach the supervisor target of `5`: `1`
- clips flagged as multi-subject-likely by the inventory harness: `4`
- harder scenarios flagged by the inventory harness: `a2`, `a3`, `b1`

Cross-clip summary from `outputs/evaluations/new_dataset_quality_pooling_phase_current/overall_evaluation_summary.json`:

- evaluated clips: `a1`, `a2`, `a3`, `b1`
- `appearance_only_pass_count = 2`
- `topology_supported_pass_count = 2`
- `topology_rescued_count = 2`
- `unknown_reuse_count = 4`
- `create_new_unknown_count = 5`
- `face_candidate_count = 12`
- `face_best_shot_selected_count = 1`
- `face_embedding_created_count = 1`

Body appearance comparison on clips that actually produced cross-camera comparisons:

- average `osnet_mean` body score: `0.5942`
- average `osnet_quality_aware` body score: `0.5939`
- average `osnet_x1_0_quality_aware` body score: `0.5719`

Important interpretation:

- the true physical `C1 -> C2` same-identity cases are still below the global sequential body threshold of `0.72`
  - `a1`: `0.6265`
  - `b1`: `0.6467`
- those cases are currently accepted by **topology-supported sequential reuse**, not by appearance alone
- appearance-only success currently happens on the easier logical copy transitions, not yet on the physical cross-view pair
- quality-aware pooling is more principled and better instrumented than mean-only pooling, but on the current local clips it is **not yet a large gain**
- the stronger OSNet x1.0 benchmark is not currently better on average than the default extractor
- `b1` created one face embedding on a face-friendly camera, but it still did not become a decisive identity anchor
- `a2` is a real blocker clip: it is paired and runnable, but it currently produces `0` entry events under the present geometry/direction/event logic

## Output Artifacts

Current phase artifacts are written under:

- `outputs/evaluations/new_dataset_inventory_phase_current/`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/`
- `outputs/evaluations/direction_validation_tracklet_phase/`
- `outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/`
- `outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet_phase_current/`

The most useful files are:

- `outputs/evaluations/new_dataset_inventory_phase_current/dataset_inventory.json`
- `outputs/evaluations/new_dataset_inventory_phase_current/dataset_inventory.md`
- `outputs/evaluations/new_dataset_inventory_phase_current/calibration_reuse_summary.json`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/overall_evaluation_summary.json`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/appearance_vs_topology_summary.json`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/face_branch_summary.json`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/qualitative_case_notes.md`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/per_clip_evaluation/a1.json`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/per_clip_evaluation/a2.json`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/per_clip_evaluation/a3.json`
- `outputs/evaluations/new_dataset_quality_pooling_phase_current/per_clip_evaluation/b1.json`
- `outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet_phase_current/body_tracklet_comparison_summary.json`
- `outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/runtime/association_logs/association_decisions.jsonl`

## Current Constraints

- the active self-recorded dataset still has only `2` physical cameras
- local paired coverage is `4` clips, not the target `5`
- the multi-subject and hard-scenario labels currently come from automated inventory sampling, not hand-labeled GT
- body appearance on the true physical `C1 -> C2` pair is still weaker than required for appearance-only acceptance at `0.72`
- topology rescue is therefore still necessary on `a1` and `b1`
- `a3` remains the strongest current failure case for cross-camera appearance robustness
- `a2` currently produces no entry events, so it blocks meaningful association evaluation for that pair
- usable face evidence is still rare on the current clips even though the branch is now audited correctly
- logical `C3/C4` remain explicit demo expansions from the 2 physical cameras
- Wildtrack benchmark assets still remain in the repo for legacy comparison and regression checks
- this repository is a thesis prototype, not a production CCTV system

## Documentation

- [docs/offline_pipeline.md](docs/offline_pipeline.md)
- [docs/new_dataset_demo.md](docs/new_dataset_demo.md)
- [docs/new_dataset_algorithmic_audit_phase_tracklet_face_topology.md](docs/new_dataset_algorithmic_audit_phase_tracklet_face_topology.md)
- [docs/live_demo_ui.md](docs/live_demo_ui.md)
- [docs/manual_scene_calibration.md](docs/manual_scene_calibration.md)
- [docs/association_runtime_config.md](docs/association_runtime_config.md)
- [docs/association_trace_logging.md](docs/association_trace_logging.md)
- [docs/association_evaluation_tuning.md](docs/association_evaluation_tuning.md)
- [docs/quantitative_evaluation.md](docs/quantitative_evaluation.md)
