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

This phase stayed at the infrastructure/debug layer. It did not add a new product surface or a new model family.

What changed:

- refreshed dataset inventory and calibration-reuse audit for the current local clips `a1`, `a2`, `a3`, `b1`
- added an `a2` overlay/debug runner so detection, tracking, direction, and event creation can be inspected frame by frame
- fixed the `a2` late-start entry failure so the clip now emits real `ENTRY_IN` events instead of dying at `TOTAL_EVENTS = 0`
- added an `a3` hard-case CV analysis runner with crop dumps, contact sheet output, preprocessing comparison, and bbox-shrink comparison
- kept `sequential.body_primary = 0.72` with no threshold rollback
- kept the existing extractor family and focused on pragmatic CV preprocessing instead:
  - `gray_world`
  - `histogram_match`
  - bbox shrink
- reran per-clip evaluation on all paired local clips and added a regression summary against the previous evaluation phase
- kept topology/time as a logged decision signal, but explicitly separated appearance-only quality from topology-supported final decisions

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
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_inventory.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --output-dir "outputs/evaluations/a2_a3_cv_phase_inventory"
```

Independent direction validation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_direction_validation.py" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --output-root "outputs/evaluations/direction_validation_tracklet_phase"
```

Per-clip evaluation across all currently paired local clips:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_evaluation.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --inventory-json ".\outputs\evaluations\a2_a3_cv_phase_inventory\dataset_inventory.json" --calibration-reuse-json ".\outputs\evaluations\a2_a3_cv_phase_inventory\calibration_reuse_summary.json" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current" --baseline-output-dir ".\outputs\evaluations\new_dataset_quality_pooling_phase_current"
```

a2 overlay debug:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_pair_debug.py" --pipeline-config ".\outputs\evaluations\a2_a3_cv_phase_current\tmp_phase_pipeline.yaml" --run-output-root ".\outputs\evaluations\a2_a3_cv_phase_current\offline_runs\a2" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current\a2_debug" --pair-id a2
```

a3 hard-case crop dump + preprocessing benchmark:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_a3_hard_case_analysis.py" --run-output-root ".\outputs\evaluations\a2_a3_cv_phase_current\offline_runs\a3" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current\a3_hard_case" --pair-id a3 --association-policy-config ".\insightface_demo_assets\runtime\config\association_policy.new_dataset_demo.yaml"
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

Current local coverage from `outputs/evaluations/a2_a3_cv_phase_inventory/dataset_inventory.json`:

- total paired clips available: `4`
- paired clips ready for evaluation: `4`
- missing paired clips to reach the supervisor target of `5`: `1`
- clips flagged as multi-subject-likely by the inventory harness: `4`
- harder scenarios flagged by the inventory harness: `a2`, `a3`, `b1`

Cross-clip summary from `outputs/evaluations/a2_a3_cv_phase_current/overall_evaluation_summary.json`:

- evaluated clips: `a1`, `a2`, `a3`, `b1`
- `appearance_only_pass_count = 4`
- `topology_supported_pass_count = 2`
- `topology_rescued_count = 2`
- `unknown_reuse_count = 6`
- `create_new_unknown_count = 14`
- `face_candidate_count = 34`
- `face_best_shot_selected_count = 2`
- `face_embedding_created_count = 2`

Current clip-level status:

- `a1`: still keeps a multi-camera unknown chain, with `C1 -> C2` accepted by topology-supported body reuse at `0.6422`
- `a2`: no longer dies at `TOTAL_EVENTS = 0`; it now emits `4` entry events, but all sequential body scores stay around `0.6001 .. 0.6007` and reuse still fails
- `a3`: still fails cross-camera reuse, but the best traditional CV combo raises the hard-case body score from `0.5071` to `0.58`
- `b1`: still keeps a multi-camera unknown chain and now produces `2` face embeddings, but the decisive physical `C1 -> C2` reuse is still topology-supported at `0.6245`

Traditional CV benchmark on `a3`:

- baseline `no_preproc_no_shrink`: average `0.4849`
- `shrink_only`: average `0.5574`
- `gray_world_shrink`: average `0.5597`
- `histogram_match_shrink`: average `0.5272`

Important interpretation:

- `a2` has been saved at the direction/event stage; the remaining blocker is appearance, not missing events
- `a3` is still the strongest hard case; simple CV preprocessing helps, but does not lift it above `0.72`
- the true physical `C1 -> C2` same-identity cases are still below the global sequential body threshold of `0.72`
- topology/time remains necessary on `a1` and `b1`, so appearance robustness is still not solved
- face branch is no longer silent, but usable embeddings remain rare and are not yet the main identity anchor

## Output Artifacts

Current phase artifacts are written under:

- `outputs/evaluations/a2_a3_cv_phase_inventory/`
- `outputs/evaluations/a2_a3_cv_phase_current/`

The most useful files are:

- `outputs/evaluations/a2_a3_cv_phase_inventory/dataset_inventory.json`
- `outputs/evaluations/a2_a3_cv_phase_inventory/calibration_reuse_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/overall_evaluation_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/regression_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/appearance_vs_topology_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/face_branch_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/per_clip_evaluation/a1.json`
- `outputs/evaluations/a2_a3_cv_phase_current/per_clip_evaluation/a2.json`
- `outputs/evaluations/a2_a3_cv_phase_current/per_clip_evaluation/a3.json`
- `outputs/evaluations/a2_a3_cv_phase_current/per_clip_evaluation/b1.json`
- `outputs/evaluations/a2_a3_cv_phase_current/a2_debug/a2_overlay_debug.mp4`
- `outputs/evaluations/a2_a3_cv_phase_current/a2_debug/a2_stage_debug_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/a2_debug/a2_root_cause_report.md`
- `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_preprocessing_benchmark.json`
- `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_bbox_shrink_benchmark.json`
- `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_hard_case_report.md`

## Current Constraints

- the active self-recorded dataset still has only `2` physical cameras
- local paired coverage is `4` clips, not the target `5`
- the multi-subject and hard-scenario labels currently come from automated inventory sampling, not hand-labeled GT
- body appearance on the true physical `C1 -> C2` pair is still weaker than required for appearance-only acceptance at `0.72`
- topology rescue is therefore still necessary on `a1` and `b1`
- `a3` remains the strongest current failure case for cross-camera appearance robustness
- `a2` no longer blocks event creation, but it still fails association because the recovered sequential body evidence stays around `0.60`
- usable face evidence is still rare on the current clips even though the branch is now audited correctly
- logical `C3/C4` remain explicit demo expansions from the 2 physical cameras
- Wildtrack benchmark assets still remain in the repo for legacy comparison and regression checks
- this repository is a thesis prototype, not a production CCTV system

## Documentation

- [docs/offline_pipeline.md](docs/offline_pipeline.md)
- [docs/new_dataset_demo.md](docs/new_dataset_demo.md)
- [docs/new_dataset_a2_a3_cv_debug_phase.md](docs/new_dataset_a2_a3_cv_debug_phase.md)
- [docs/new_dataset_algorithmic_audit_phase_tracklet_face_topology.md](docs/new_dataset_algorithmic_audit_phase_tracklet_face_topology.md)
- [docs/live_demo_ui.md](docs/live_demo_ui.md)
- [docs/manual_scene_calibration.md](docs/manual_scene_calibration.md)
- [docs/association_runtime_config.md](docs/association_runtime_config.md)
- [docs/association_trace_logging.md](docs/association_trace_logging.md)
- [docs/association_evaluation_tuning.md](docs/association_evaluation_tuning.md)
- [docs/quantitative_evaluation.md](docs/quantitative_evaluation.md)
