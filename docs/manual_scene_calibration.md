# Manual Scene Calibration

Manual ROI and scene calibration is the required runtime path for this repository.

The old auto/inferred ROI flow is deprecated for runtime use and is no longer allowed to silently drive event creation.

## Why

The previous defaulted ROI/zone behavior was too loose for:

- stable `IN` event creation
- correct zone/subzone assignment
- map-aware cross-camera association
- compute reduction in the live path

The current runtime therefore requires an explicit scene calibration file before offline or live event generation.

## Active Config File

Current active New Dataset calibration:

- `insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml`

Legacy Wildtrack calibration files still exist for regression/reference, but they are not the active default dataset path.

The active calibration stores normalized coordinates for:

- `processing_roi`
- `entry_line`
- `zones`
- `subzones`
- direction filter history settings
- per-camera anchor point mode

Coordinates are normalized to frame width and height so the same file can be reused across clips from the same source camera.

## Calibration Reuse By Source Camera

For the active self-recorded dataset, calibration should be reused by **source camera**, not redrawn per clip.

Current rule:

- all clips in `New Dataset/Camera 1` reuse the Camera 1 calibration geometry
- all clips in `New Dataset/Camera 2` reuse the Camera 2 calibration geometry

This is the correct behavior when these remain materially unchanged:

- camera position
- camera angle
- background geometry
- normalized coordinate space / compatible aspect ratio

Current inventory audit confirms that local clips `a1`, `a2`, `a3`, and `b1` all reuse the existing calibration successfully.

This phase kept that rule unchanged:

- `a2` was fixed at the direction/event layer without redrawing calibration
- `a3` was analyzed through crop preprocessing and bbox shrink without creating clip-specific calibration duplicates

## Runtime Behavior

Offline and live entrypoints require manual calibration:

- offline: `insightface_demo_assets/runtime/run_offline_multicam_pipeline.py`
- live compatibility path: `insightface_demo_assets/runtime/run_face_resolution_demo.py`
- demo server / preview path: `insightface_demo_assets/runtime/run_live_event_demo_server.py`

If the manual calibration config is missing or invalid:

- runtime stops with a clear error
- the old bad ROI fallback is not used

The calibration UI is the only preview-oriented exception:

- it may open in preview mode and let you draw/save calibration
- runtime ingestion still requires a valid saved config

For the current local clips, the correct workflow is:

1. reuse the same source-camera calibration for `a1`, `a2`, `a3`, `b1`
2. only redraw if the source camera geometry changes materially
3. debug failures first through overlay, track dumps, zone/subzone audit, or crop analysis before touching calibration

## Calibration UI

The existing live demo server also serves:

- `/calibration.html`

Capabilities:

- choose camera
- load preview frame
- draw/edit/delete processing ROI
- draw/edit/delete entry line and IN-side anchor
- draw/edit/delete zones
- draw/edit/delete subzones
- save config
- reload config
- reset per-camera config

## Direction Stabilization

`IN` is no longer decided from a single-frame tripwire crossing alone.

The current runtime direction decision combines:

- line crossing
- motion history over a configurable window
- inward momentum
- inside ratio on the protected side
- optional zone/subzone transition cues

Direction metadata is written into event audit rows, for example:

- `direction_reason`
- `direction_history_points`
- `direction_momentum_px`
- `direction_inside_ratio`

## ROI Masking

The pipeline uses the calibrated `processing_roi` as a practical mask/filter:

- optionally mask outside the polygon before detection
- keep point-in-polygon filtering on emitted detections/tracks

This reduces wasted compute outside the protected region without changing the association core.

Current phase artifacts that validate calibration reuse in practice:

- `outputs/evaluations/a2_a3_cv_phase_inventory/calibration_reuse_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/a2_debug/a2_overlay_debug.mp4`
- `outputs/evaluations/a2_a3_cv_phase_current/a2_debug/a2_stage_debug_summary.json`
