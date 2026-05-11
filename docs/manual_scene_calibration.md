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

## OpenCV Calibration Tool

For a new camera source, use the standalone OpenCV tool when you need a quick
mouse-driven calibration without starting the web UI:

```powershell
cd /d "<repo-root>"
& ".\.venv_insightface_demo\Scripts\python.exe" `
  ".\tools\calibrate_camera.py" `
  --camera C1 `
  --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\a1.mp4" `
  --output-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.custom.yaml" `
  --frame-index 0
```

Alternative direct module path:

```powershell
& ".\.venv_insightface_demo\Scripts\python.exe" `
  ".\insightface_demo_assets\runtime\tools\calibrate_camera.py" `
  --camera C2 `
  --source "0" `
  --output-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.custom.yaml"
```

Supported source types are whatever OpenCV can open:

- image file
- video file
- webcam index such as `0`
- RTSP / HTTP stream URL

Useful options:

- `--frame-index N` selects a frame from a video.
- `--time-sec S` selects a video timestamp.
- `--base-config existing.yaml` merges into an existing scene calibration.
- `--per-camera-json C1.json` writes a simple per-camera JSON export.
- `--anchor-point-mode bottom_center|center_center` sets the ROI test anchor.
- `--no-default-zone` disables the generated zone-from-ROI helper.

Mouse / keyboard controls:

- left click adds a point
- right click or Enter commits the ROI polygon
- after ROI commit, click entry line point 1, entry line point 2, then the IN-side point
- `u` undoes the last point
- `r` resets the current shape
- `s` saves once ROI and entry line are complete
- `q` or Esc quits without saving

The overlay shows the detection ROI, the entry line, and an arrow pointing to the
clicked IN side.

### Output Format

The main output is runtime-compatible scene calibration:

```yaml
scene_calibration:
  coordinate_space: normalized
  cameras:
    C1:
      processing_roi:
        polygon: [[...], [...], [...]]
      entry_line:
        points: [[x1, y1], [x2, y2]]
        in_side_point: [xin, yin]
      zones:
        - zone_id: c1_entry_main
          zone_type: entry
          polygon: ...
```

The per-camera JSON export is for supervisor/debug review and contains pixel
coordinates plus the explicit direction vector:

```json
{
  "camera_id": "C1",
  "resolution": [1418, 720],
  "detection_roi": [[x1, y1], [x2, y2]],
  "entry_line": {
    "p1": [xa, ya],
    "p2": [xb, yb],
    "in_side_point": [xi, yi],
    "in_direction_vector": [dx, dy]
  }
}
```

The backend still uses `entry_line.in_side_point` for line-side tests. The
`in_direction_vector` is exported as an explicit audit field: movement with a
positive dot product against this vector moves toward the facility side.

### Pipeline Integration

The generated YAML/JSON can be passed directly to the existing runtime:

```powershell
& ".\.venv_insightface_demo\Scripts\python.exe" `
  ".\insightface_demo_assets\runtime\run_live_event_demo_server.py" `
  --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.custom.yaml"
```

For the offline logical demo, point the pipeline config's
`scene_calibration_config` at the generated file. `C3` and `C4` may reuse/copy
the physical source-camera geometry from `C1` and `C2`, but that relationship
must stay explicit in config/docs because they are logical replay cameras.

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
