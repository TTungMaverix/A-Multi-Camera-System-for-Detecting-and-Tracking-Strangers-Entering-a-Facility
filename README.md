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
- `b2`

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

## P0 Repair Status

The current P0 branch addresses supervisor-blocking issues that must be solved before merging into `main`:

- C4 `ENTRY_IN` must be produced by direction analysis, not by API/UI event cloning.
- Video decode and overlay must run in background workers; HTTP endpoints must read buffered frames.
- The active Known DB is the facility folder under `New Dataset/Known ID`.
- Face matching keeps embedding cosine as the primary signal and adds aligned grayscale face similarity as auxiliary evidence.
- Re-ID decisions expose `FaceScore`, `BodyScore`, `TimeScore`, `TopologyScore`, `FinalScore`, and threshold.
- API/OpenAPI docs now include camera state, MJPEG streaming, association decisions, and artifact-backed endpoints.

P0 validation artifacts from the current branch:

- `outputs/evaluations/p0_c4_direction_debug/c4_direction_debug.json`
- `outputs/evaluations/p0_c4_direction_debug/c4_direction_overlay_frame.png`
- `outputs/evaluations/p0_c4_direction_debug/c4_track_coordinates.csv`
- `outputs/evaluations/known_facility_db/known_db_build_summary.json`
- `outputs/evaluations/server_fps_benchmark/server_runtime_summary.json`
- `outputs/evaluations/server_fps_benchmark/endpoint_smoke_results.json`

Observed P0 validation snapshot:

- C4 event exists naturally in `entry_in_events.csv`: `IN_C4_C4_1_00000331`
- C4 direction accept mode: `entry_inferred_from_inside_roi_track_start`
- known DB build loaded `3` identities and created `12` embeddings / aligned grayscale crops
- worker-buffer benchmark reached average `14.885 FPS` across 4 streams against a `15 FPS` target

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

Build active facility Known DB:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_build_known_facility_db.py" --project-root "." --known-root "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID" --manifest-csv ".\insightface_demo_assets\known_face_facility_manifest.csv" --embeddings-csv ".\insightface_demo_assets\runtime\known_face_facility_embeddings.csv" --summary-json ".\outputs\evaluations\known_facility_db\known_db_build_summary.json"
```

Debug C4 direction:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_pair_debug.py" --run-output-root ".\outputs\evaluations\p0_repair_eval_a1\offline_runs\a1" --output-dir ".\outputs\evaluations\p0_c4_direction_debug" --pair-id a1 --camera-id C4
```

Draw ROI and entry line for a new camera:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" --camera C1 --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\a1.mp4" --output-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.custom.yaml" --frame-index 0
```

Controls:

- left click: add ROI / line point
- right click or Enter: commit ROI
- entry line uses three clicks: `p1`, `p2`, then IN-side point
- `u`: undo
- `r`: reset current shape
- `s`: save
- `q` / Esc: quit

Run live demo server with buffered MJPEG streams:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_live_event_demo_server.py" --host 127.0.0.1 --port 8765 --project-root "." --output-root ".\outputs\evaluations\p0_repair_eval_a1\offline_runs\a1" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --stream-target-fps 15
```

Benchmark server frame workers:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_server_fps_benchmark.py" --project-root "." --output-root ".\outputs\evaluations\p0_repair_eval_a1\offline_runs\a1" --duration-sec 6 --target-fps 15 --summary-json ".\outputs\evaluations\server_fps_benchmark\server_runtime_summary.json"
```

Regression tests:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" -m pytest tests -q
```

## Current Validation Snapshot

Current local coverage from `outputs/evaluations/p0_repair_eval_all/overall_evaluation_summary.json`:

- evaluated paired clips: `5`
- paired clips: `a1`, `a2`, `a3`, `b1`, `b2`
- missing paired clips to reach the supervisor target of `5`: `0`
- multi-subject ground truth still needs manual verification

Cross-clip summary from `outputs/evaluations/p0_repair_eval_all/overall_evaluation_summary.json`:

- evaluated clips: `a1`, `a2`, `a3`, `b1`, `b2`
- `appearance_only_pass_count = 4`
- `topology_supported_pass_count = 2`
- `topology_rescued_count = 2`
- `unknown_reuse_count = 6`
- `create_new_unknown_count = 16`
- `face_candidate_count = 34`
- `face_best_shot_selected_count = 2`
- `face_embedding_created_count = 2`

Current clip-level status:

- `a1`: still keeps a multi-camera unknown chain, with `C1 -> C2` accepted by topology-supported body reuse at `0.6422`
- `a2`: no longer dies at `TOTAL_EVENTS = 0`; it now emits `4` entry events, but all sequential body scores stay around `0.6001 .. 0.6007` and reuse still fails
- `a3`: still fails cross-camera reuse, but the best traditional CV combo raises the hard-case body score from `0.5071` to `0.58`
- `b1`: still keeps a multi-camera unknown chain and now produces `2` face embeddings, but the decisive physical `C1 -> C2` reuse is still topology-supported at `0.6245`
- `b2`: now appears in the paired local dataset and evaluates to `2` events, `0` reuse, and `0.4508` average quality-aware body score

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
- `outputs/evaluations/p0_repair_eval_all/`
- `outputs/evaluations/p0_c4_direction_debug/`
- `outputs/evaluations/known_facility_db/`
- `outputs/evaluations/server_fps_benchmark/`

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
- `outputs/evaluations/p0_repair_eval_all/per_clip_table.json`
- `outputs/evaluations/p0_repair_eval_all/overall_evaluation_summary.json`
- `outputs/evaluations/p0_c4_direction_debug/c4_direction_debug.json`
- `outputs/evaluations/p0_c4_direction_debug/c4_direction_overlay_frame.png`
- `outputs/evaluations/known_facility_db/known_db_build_summary.json`
- `outputs/evaluations/server_fps_benchmark/server_runtime_summary.json`

## Current Constraints

- the active self-recorded dataset still has only `2` physical cameras
- local paired coverage is now `5` clips, but multi-subject ground truth still needs manual verification
- the hard-scenario labels currently come from automated inventory sampling, not hand-labeled GT
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
- [docs/project_constraints.md](docs/project_constraints.md)
- [docs/direction_detector_debug.md](docs/direction_detector_debug.md)
- [docs/known_face_db.md](docs/known_face_db.md)
- [docs/api/openapi.yaml](docs/api/openapi.yaml)
