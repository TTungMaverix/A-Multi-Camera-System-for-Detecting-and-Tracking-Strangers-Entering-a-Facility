# Lightweight Live Demo UI

The UI in this repository is not meant to be a separate product. It serves two practical jobs:

1. calibrate ROI and direction geometry faster
2. inspect real event/timeline outputs during debugging and during the thesis defense

## Runtime Pieces

Server:

- `insightface_demo_assets/runtime/run_live_event_demo_server.py`

Wrappers:

- `run_live_event_demo_server_b1.cmd`
- `run_live_event_demo_server.ps1` (legacy)

Static pages:

- `insightface_demo_assets/runtime/web_demo/index.html`
- `insightface_demo_assets/runtime/web_demo/calibration.html`

## Current Data Sources

The server can read both live-run and offline-run artifacts.

Live/simulated real-time artifacts:

- `outputs/live_runs/<run_name>/events/latest_events.json`
- `outputs/live_runs/<run_name>/summaries/live_pipeline_summary.json`
- `outputs/live_runs/<run_name>/events/simulated_realtime_trace.jsonl`

Offline 4-camera ROI benchmark artifacts:

- `outputs/offline_runs/<run_name>/events/latest_events.json`
- `outputs/offline_runs/<run_name>/timelines/unknown_identity_timeline.json`
- `outputs/offline_runs/<run_name>/summaries/cross_camera_handoff_summary.json`

The current PowerShell wrapper is configured to point at the official 4-camera ROI benchmark output root.

## Timeline View

The main page now uses real timeline data, not a mock list.

For each Unknown identity it shows:

- identity label
- first seen / last seen
- total appearances
- camera sequence
- per-appearance timestamp
- best-shot body/head crops
- modality used
- decision reason
- zone/subzone context

The main interaction is:

1. click an event or timeline card
2. inspect the ordered appearance history
3. confirm the camera handoff sequence and associated best-shots

## Calibration Tool

The calibration page is now the practical ROI tool for this phase.

Key improvements:

- dark compact layout
- larger preview canvas
- clean frame reload instead of reusing already scribbled previews
- per-shape commit instead of forcing a full reset
- undo last draft point
- delete selected shape
- reload existing shapes for editing
- shape list panel
- dropdown presets instead of heavy manual typing

Supported geometry types:

- processing ROI polygon
- entry line
- zone polygon
- subzone polygon

## Endpoints

- `/`
- `/index.html`
- `/calibration.html`
- `/api/camera-state`
- `/api/camera-config`
- `/api/camera-frame?camera_id=...`
- `/api/latest-events`
- `/api/reid-handoffs`
- `/api/summary`
- `/api/timeline`
- `/api/calibration/state`
- `/api/calibration/preview?camera_id=...`
- `/api/calibration/save`
- `/api/calibration/reset`
- `/artifact?path=...`

## Supervisor Demo Command

```cmd
cd /d "<repo-root>"
set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
run_live_event_demo_server_b1.cmd
```

Default URL:

- `http://127.0.0.1:8765`

Recommended demo pair for supervisor playback:

- `b1`

Reason:

- the current regression summary already shows `b1` has real event/timeline content suitable for the UI demo
- `b2` currently has zero events and is not suitable as the default live UI showcase

## Current Playback Behavior

- playback is sequential: `C1 -> gap -> C2 -> gap -> C3 -> gap -> C4`
- default segment length in the `b1` wrapper: `12` seconds
- default travel gap in the `b1` wrapper: `8` seconds
- UI only refreshes the active camera feed
- inactive camera tiles stay visible at equal size and switch to standby

## Current Limitation

This UI is still a thesis tool:

- it is not a production monitoring dashboard
- it depends on artifacts already written by the pipeline
- if backend association is wrong, the UI will expose that wrongness rather than hide it
- if the server still decodes preview frames from source clips internally, that cost remains server-side; this phase only ensures the client refreshes the active camera tile
