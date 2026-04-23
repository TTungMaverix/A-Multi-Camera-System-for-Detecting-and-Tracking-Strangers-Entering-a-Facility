# Manual Scene Calibration

Manual ROI and scene calibration is now the required runtime path for this repository.

The old auto/inferred ROI flow is deprecated for runtime use and is no longer allowed to silently drive event creation.

## Why

The previous defaulted ROI/zone behavior was too loose for:

- stable `IN` event creation
- correct zone/subzone assignment
- overlap-aware cross-camera association
- compute reduction in the live path

This phase replaces that with an explicit scene calibration file that the runtime must load before offline or live event generation.

## Config File

Current demo calibration:

- `insightface_demo_assets/runtime/config/manual_scene_calibration.wildtrack.json`

It stores normalized coordinates for:

- `processing_roi`
- `entry_line`
- `zones`
- `subzones`
- direction filter history settings

Coordinates are normalized to frame width and height so the same file can be reloaded on preview frames and runtime feeds.

## Runtime Behavior

Offline and live entrypoints now require manual calibration:

- offline: `insightface_demo_assets/runtime/run_offline_multicam_pipeline.py`
- live: `insightface_demo_assets/runtime/run_live_multicam_demo.py`
- face-resolution compatibility path: `insightface_demo_assets/runtime/run_face_resolution_demo.py`

If the manual calibration config is missing or invalid:

- runtime stops with a clear error
- the old bad ROI fallback is not used

The web calibration tool is the only preview-oriented exception:

- it may open in preview mode and let you draw/save calibration
- but runtime ingestion still requires a valid saved config

## Calibration UI

The existing live demo server now also serves:

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

The live pipeline now uses the calibrated `processing_roi` as a practical mask/filter:

- mask outside the polygon before lightweight detection
- keep point-in-polygon filtering on emitted detections/tracks

This reduces wasted compute outside the protected region without changing the association core.
