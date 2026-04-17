# Lightweight Live Demo UI

The UI in this repository is not meant to be a separate product. It serves two practical jobs:

1. calibrate ROI and direction geometry faster
2. inspect real event/timeline outputs during debugging and during the thesis defense

## Runtime Pieces

Server:

- `insightface_demo_assets/runtime/run_live_event_demo_server.py`

Wrapper:

- `run_live_event_demo_server.ps1`

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
- `/api/latest-events`
- `/api/summary`
- `/api/timeline`
- `/api/calibration/state`
- `/api/calibration/preview?camera_id=...`
- `/api/calibration/save`
- `/api/calibration/reset`
- `/artifact?path=...`

## Command

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_live_event_demo_server.ps1"
```

Default URL:

- `http://127.0.0.1:8765`

## Current Limitation

This UI is still a thesis tool:

- it is not a production monitoring dashboard
- it depends on artifacts already written by the pipeline
- if backend association is wrong, the UI will expose that wrongness rather than hide it
