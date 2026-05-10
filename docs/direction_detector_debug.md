# Direction Detector Debugging

The P0 repair removed demo-layer C4 event fabrication and requires C4 `ENTRY_IN` to come from the direction detector.

## Debug Command

```powershell
& "D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe" `
  insightface_demo_assets/runtime/run_new_dataset_pair_debug.py `
  --run-output-root outputs/evaluations/p0_repair_eval_a1/offline_runs/a1 `
  --output-dir outputs/evaluations/p0_c4_direction_debug `
  --pair-id a1 `
  --camera-id C4
```

## Required Artifacts

- `outputs/evaluations/p0_c4_direction_debug/c4_track_coordinates.csv`
- `outputs/evaluations/p0_c4_direction_debug/c4_direction_debug.json`
- `outputs/evaluations/p0_c4_direction_debug/c4_direction_overlay_frame.png`
- `outputs/evaluations/p0_c4_direction_debug/c4_direction_overlay.mp4`

## Direction Acceptance Modes

- `line_cross_in`: primary line crossing.
- `roi_entry_transition`: outside-to-inside ROI transition.
- `entry_inferred_from_inside_roi_track_start`: clipped-track fallback. This is valid only inside direction analysis and must include persistence/motion/ROI evidence.

## P0 C4 Evidence

The repaired a1 run generated a natural C4 event:

- `event_id`: `IN_C4_C4_1_00000331`
- `camera_id`: `C4`
- `direction`: `IN`
- `direction_accept_mode`: `entry_inferred_from_inside_roi_track_start`
- `direction_reason`: source starts inside the entry ROI with sufficient persistence and inward evidence.

This event appears in `events/entry_in_events.csv`, so the API/UI does not need to clone a C2 event into C4.
