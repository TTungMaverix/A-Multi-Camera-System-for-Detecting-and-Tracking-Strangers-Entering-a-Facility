# Camera Subzone Config

Subzones are the next layer below camera-level zones.

They let the association core reason about:

- which side of a camera view a stranger came from
- which interior band or overlap band they were seen in
- whether a transition is plausible for a specific entry/exit path

## Where Subzones Live

Subzones are currently stored inside:

- [manual_scene_calibration.new_dataset_demo.yaml](../insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml)
- [camera_transition_map.example.yaml](../insightface_demo_assets/runtime/config/camera_transition_map.example.yaml)

The scene calibration file owns camera-local ROI, zone, and subzone geometry.
The transition map owns directed route constraints such as allowed entry/exit
zone IDs and subzone IDs.

The OpenCV calibration tool writes camera-local `zones` into the scene
calibration file when it generates a default zone from the processing ROI. That
zone is used by event creation and spatial assignment. Directed transition
constraints such as `allowed_entry_subzones` and `allowed_exit_subzones` still
belong in the camera transition map.

Tool command:

```powershell
& ".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" `
  --camera C1 `
  --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\a1.mp4" `
  --output-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.custom.yaml"
```

By default, the tool creates:

- `processing_roi.polygon`
- `entry_line.points`
- `entry_line.in_side_point`
- a default zone named `<camera>_entry_main` whose polygon matches the ROI

Subzones remain a second pass when a dataset needs finer route constraints.
The OpenCV tool can now create minimal review-ready subzone placeholders with:

```powershell
& ".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" `
  --camera C1 `
  --source "D:\DO AN TOT NGHIEP\New Dataset\Camera 1\a1.mp4" `
  --output-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.custom.yaml" `
  --with-default-subzones
```

That flag writes:

- `<camera>_entry_band`: projected from the entry line toward the clicked IN side.
- `<camera>_interior`: a second projected band farther inside the ROI.

Both are marked `placeholder: true`. Treat them as a starting schema and visual
review target, not as final route evidence until checked against the camera view.

## Camera-Level Fields

Each camera can define:

- `default_subzone_id`
- `subzones`

Each subzone can define:

- `subzone_id`
- `parent_zone_id`
- `subzone_type`
- `polygon`
- `priority`
- `allowed_transitions`
- `placeholder`
- `description`

`priority` is used when more than one subzone polygon matches the same point.

## Transition-Level Fields

Each directed transition can define:

- `allowed_exit_subzones`
- `allowed_entry_subzones`

These are optional. If they are omitted, the filter falls back to zone-level logic.

For logical replay cameras:

- `C3` should derive geometry from the same source-camera calibration as `C1`.
- `C4` should derive geometry from the same source-camera calibration as `C2`.
- Keep the derived relationship explicit in config/docs. Do not create hidden
  per-clip or per-replay calibration unless the source camera geometry changes.

## Runtime Behavior

During event generation:

1. use the current event foot point when available
2. assign `zone_id` from zone polygons
3. assign `subzone_id` from subzone polygons under that zone when possible
4. if no subzone matches, fall back to the camera default subzone
5. record audit fields:
   - `subzone_id`
   - `subzone_type`
   - `subzone_reason`
   - `subzone_fallback_used`

During association:

1. topology must allow the camera pair
2. time must fit the relation window
3. zone must be compatible if zone constraints exist
4. subzone must be compatible if subzone constraints exist
5. if subzone data is missing, the filter can fall back according to policy

## Current Wildtrack Example

The current example config gives each camera:

- a default zone
- a default subzone
- at least one approximate entry/overlap subzone
- at least one approximate exit/interior subzone

Some follow-up camera subzones are marked as placeholders because Wildtrack is not a clean facility-entry dataset. This is intentional and keeps the design honest while still making the map-aware logic testable.
