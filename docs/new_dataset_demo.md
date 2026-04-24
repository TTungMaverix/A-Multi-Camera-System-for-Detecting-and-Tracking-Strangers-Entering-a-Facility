# New Dataset Demo Profile

## Purpose

This document explains how the self-recorded New Dataset is connected to the existing thesis pipeline without rewriting the core algorithmic blocks.

## Physical Dataset Layout

Current physical folders:

- `New Dataset/Camera 1`
- `New Dataset/Camera 2`

Current local paired stems:

- `a1`
- `a2`
- `a3`
- `b1`

Current pairing rule:

- `Camera 1/<stem>.mp4` pairs with `Camera 2/<stem>.mp4`

The runtime adapter discovers clip pairs by stem intersection across the two physical camera folders. It does not hardcode the runtime to `a1`.

Current inventory snapshot:

- total paired clips available: `4`
- paired clips ready for evaluation: `4`
- missing paired clips to reach target `5`: `1`

## Logical 4-Camera Demo Expansion

The thesis scope stays at 4-camera demonstration level, but the active dataset currently has only 2 physical cameras.

The repo therefore uses explicit logical demo streams:

- `C1` -> physical Camera 1, offset `0s`
- `C2` -> physical Camera 2, offset `10s`
- `C3` -> replayed logical copy of physical Camera 1, offset `20s`
- `C4` -> replayed logical copy of physical Camera 2, offset `30s`

This is a documented demo adapter for scope alignment only.

## Camera Geometry Assumptions

- Camera 1:
  - top-down-ish diagonal view
  - face and full body are usually visible
  - ROI anchor mode defaults to `bottom_center`
- Camera 2:
  - more eye-level diagonal view
  - partial-body / upper-body detection is more common
  - ROI anchor mode defaults to `center_center`

The point-in-polygon math itself is unchanged. Only the anchor point used before the polygon test is configurable per camera.

## Calibration Reuse

Calibration is tied to the **source camera**, not to individual clips.

That means:

- all clips from `New Dataset/Camera 1` reuse the Camera 1 calibration geometry
- all clips from `New Dataset/Camera 2` reuse the Camera 2 calibration geometry

This is the correct policy as long as these stay materially unchanged:

- camera position
- camera angle
- background geometry
- resolution / aspect ratio within compatible tolerance

The current calibration reuse audit confirms:

- `a1`, `a2`, `a3`, `b1` all reuse the existing Camera 1 calibration
- `a1`, `a2`, `a3`, `b1` all reuse the existing Camera 2 calibration

No per-clip redraw is needed right now.

## Topology

The New Dataset is treated as **non-overlap oriented**.

Association therefore emphasizes:

- reachable camera pairs
- min/max travel-time windows
- topology-supported sequential reuse
- face/body evidence layered on top of map/time constraints

Current default travel-time assumptions:

- `C1 -> C2`: `5s .. 30s`
- `C2 -> C3`: `5s .. 30s`
- `C3 -> C4`: `5s .. 30s`

## Files

- dataset profile:
  - `insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml`
- scene calibration:
  - `insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml`
- topology:
  - `insightface_demo_assets/runtime/config/camera_transition_map.new_dataset_demo.yaml`
- policy:
  - `insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml`
- offline demo config:
  - `insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml`
- inventory runner:
  - `insightface_demo_assets/runtime/run_new_dataset_inventory.py`
- per-clip evaluation runner:
  - `insightface_demo_assets/runtime/run_new_dataset_evaluation.py`
- body tracklet comparison runner:
  - `insightface_demo_assets/runtime/run_body_tracklet_evaluation.py`

## Current Status

This phase does not claim that appearance is already solved. It does claim that the repo now has an honest dataset audit and evaluation harness for the clips that exist locally.

Current local behavior:

- `a1`:
  - keeps a single unknown chain across `C1 -> C2 -> C3 -> C4`
  - physical `C1 -> C2` still needs topology-supported accept at `0.6422`
- `a2`:
  - paired and runnable
  - no longer dies at `TOTAL_EVENTS = 0`
  - now emits `4` entry events recovered by the late-start inside-entry fallback
  - still fails cross-camera reuse because the recovered sequential body scores remain around `0.6001 .. 0.6007`
- `a3`:
  - current hardest failure case
  - no cross-camera reuse
  - traditional CV preprocessing helps, but best score still only reaches `0.58`
- `b1`:
  - keeps a single unknown chain across `C1 -> C2 -> C3 -> C4`
  - creates `2` face embeddings on the face-friendlier path
  - physical `C1 -> C2` still needs topology-supported accept at `0.6245`

Useful commands:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_inventory.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --output-dir "outputs/evaluations/a2_a3_cv_phase_inventory"
```

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_evaluation.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --inventory-json ".\outputs\evaluations\a2_a3_cv_phase_inventory\dataset_inventory.json" --calibration-reuse-json ".\outputs\evaluations\a2_a3_cv_phase_inventory\calibration_reuse_summary.json" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current" --baseline-output-dir ".\outputs\evaluations\new_dataset_quality_pooling_phase_current"
```

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_pair_debug.py" --pipeline-config ".\outputs\evaluations\a2_a3_cv_phase_current\tmp_phase_pipeline.yaml" --run-output-root ".\outputs\evaluations\a2_a3_cv_phase_current\offline_runs\a2" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current\a2_debug" --pair-id a2
```

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_a3_hard_case_analysis.py" --run-output-root ".\outputs\evaluations\a2_a3_cv_phase_current\offline_runs\a3" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current\a3_hard_case" --pair-id a3 --association-policy-config ".\insightface_demo_assets\runtime\config\association_policy.new_dataset_demo.yaml"
```
