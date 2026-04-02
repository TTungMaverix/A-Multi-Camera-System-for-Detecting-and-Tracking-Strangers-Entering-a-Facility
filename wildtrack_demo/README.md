# Wildtrack Demo Prep

This folder prepares a 4-camera Wildtrack subset for the graduation-project demo.
It is only the data-prep layer that feeds the architecture shown in:

- `D:\ĐỒ ÁN TỐT NGHIỆP\Important\OverallDiagram.drawio.png`
- `D:\ĐỒ ÁN TỐT NGHIỆP\Important\WorkFlow.drawio.png`

## Alignment With Your Diagrams

Overall diagram alignment:

- `C3`, `C5`, `C6`, `C7` act as the temporary 4-camera input set
- this prep keeps per-camera processing separated first, then exports assets for the central identity association stage
- `entry_in_events` and `identity_resolution_queue` are the handoff point from per-camera CV to the central known/unknown decision core
- `camera_topology` in config is the temporary replacement for `camera_map / travel_time / ROI / direction rules`
- `best_head_crop` and `best_body_crop` are the inputs for known DB matching, fallback body feature matching and unknown profile update

Workflow alignment:

1. Keep full frames and apply fixed ROI logic per selected camera.
2. Detect valid person presence from Wildtrack annotations inside ROI.
3. Trigger `ENTRY_IN` only when the foot point crosses the configured line toward the `IN` side.
4. Export `best_head_crop` for face quality / face embedding / known DB matching.
5. Export `best_body_crop` as fallback body feature input when face is weak or unmatched.
6. Send every entry event into `identity_resolution_queue.csv` for `known` vs `unknown` decision.
7. Use overlap topology as the current spatio-temporal candidate filter.
8. Continue cross-camera reasoning with `C3/C5/C6/C7` and keep logs for dashboard/timeline later.

## Current Camera Roles

- `C5`, `C6`: entry cameras with fixed ROI and `ENTRY_IN` line crossing
- `C3`, `C7`: follow-up cameras for cross-camera continuity
- keep original full frames and use ROI logic instead of physically cropping the dataset first

## Main Files

- `wildtrack_demo_config.json`: ROI, entry line, camera roles, overlap topology
- `export_wildtrack_demo.ps1`: convert Wildtrack annotations into demo-ready assets
- `output\tracks\*.csv`: filtered per-camera track rows inside the configured ROI
- `output\events\entry_in_events.csv`: entry events triggered by line crossing
- `output\events\identity_resolution_queue.csv`: handoff file for known/unknown face matching
- `output\summary\global_gt_summary.csv`: debug/evaluation summary of the same person across selected cameras
- `output\overlays\*.png`: visual check for ROI, line and crop placement

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\wildtrack_demo\export_wildtrack_demo.ps1
```

## Current Export Snapshot

- filtered track rows: `11352`
- `ENTRY_IN` events: `51`
- identity-resolution queue rows: `51`
- event split: `C5=38`, `C6=13`
