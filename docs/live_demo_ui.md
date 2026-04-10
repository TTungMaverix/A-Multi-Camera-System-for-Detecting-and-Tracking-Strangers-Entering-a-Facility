# Lightweight Live Demo UI

Phase H adds a minimal web demo for the thesis defense.

It is intentionally simple:

- no database
- no authentication
- no production dashboard stack
- no analytics backend

The UI only needs to show:

- a new event arrived
- which camera it came from
- whether the identity is `known` or `unknown`
- which ID was assigned
- which zone/subzone it belongs to
- which snapshot was stored

The same lightweight server now also provides a calibration page so ROI/zone/subzone geometry can be edited without adding a second demo app.

## Runtime Pieces

Server script:

- `insightface_demo_assets/runtime/run_live_event_demo_server.py`

PowerShell wrapper:

- `run_live_event_demo_server.ps1`

Static page:

- `insightface_demo_assets/runtime/web_demo/index.html`
- `insightface_demo_assets/runtime/web_demo/calibration.html`

## Data Source

The UI reads the outputs created by the live pipeline:

- `outputs/live_runs/<run_name>/events/latest_events.json`
- `outputs/live_runs/<run_name>/summaries/live_pipeline_summary.json`

Snapshot images are served through:

- `/artifact?path=<absolute_or_repo_relative_path>`

The server validates that the requested file still stays inside the project root before serving it.

## Endpoints

- `/` or `/index.html`
- `/calibration.html`
- `/api/latest-events`
- `/api/summary`
- `/api/calibration/state`
- `/api/calibration/preview?camera_id=...`
- `/api/calibration/save`
- `/api/calibration/reset`
- `/artifact?path=...`

## Commands

Start the live pipeline first:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
powershell -ExecutionPolicy Bypass -File ".\run_live_multicam_demo.ps1"
```

Then start the demo UI:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
powershell -ExecutionPolicy Bypass -File ".\run_live_event_demo_server.ps1"
```

Default URL:

- `http://127.0.0.1:8765`

## Current Limitation

This is a defense-oriented UI, not a finished monitoring product.

The live page reflects whatever the current live pipeline emits. If the live ingestion is run through file replay or only 2 cameras, the page will reflect that exact setup.
