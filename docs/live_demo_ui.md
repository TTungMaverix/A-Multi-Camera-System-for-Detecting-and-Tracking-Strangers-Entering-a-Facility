# Lightweight Live Demo UI

The UI in this repository is not meant to be a separate product. It serves two practical jobs:

1. calibrate ROI and direction geometry faster
2. inspect real event/timeline outputs during debugging and during the thesis defense

## Runtime Pieces

Server:

- `insightface_demo_assets/runtime/run_live_event_demo_server.py`
- `insightface_demo_assets/runtime/run_server_fps_benchmark.py`

Wrapper:

- `run_live_event_demo_server.ps1`

Static pages:

- `insightface_demo_assets/runtime/web_demo/index.html`
- `insightface_demo_assets/runtime/web_demo/calibration.html`

## Current Data Sources and Streaming Model

The server can read both live-run and offline-run artifacts. In the P0 runtime it also starts background replay workers that decode and annotate frames independently of the HTTP request path.

Live/simulated real-time artifacts:

- `outputs/live_runs/<run_name>/events/latest_events.json`
- `outputs/live_runs/<run_name>/summaries/live_pipeline_summary.json`
- `outputs/live_runs/<run_name>/events/simulated_realtime_trace.jsonl`

Offline 4-camera ROI benchmark artifacts:

- `outputs/offline_runs/<run_name>/events/latest_events.json`
- `outputs/offline_runs/<run_name>/timelines/unknown_identity_timeline.json`
- `outputs/offline_runs/<run_name>/summaries/cross_camera_handoff_summary.json`

The current PowerShell wrapper is configured to point at the official 4-camera ROI benchmark output root.

Frame serving:

- background workers decode video and draw overlays
- each worker writes the latest JPEG and metadata into an in-memory `FrameBufferHub`
- `/api/camera-frame` returns the latest buffered JPEG
- `/api/camera-stream` and `/stream/camera/{camera_id}.mjpg` stream MJPEG from the same buffer
- the browser uses MJPEG `<img>` streams for camera tiles and polls `/api/camera-state` only for metadata

This avoids the old bottleneck where `/api/camera-frame` performed video seeking and preview rendering per HTTP request.

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
- `/api/reid-handoffs`
- `/api/association-decisions`
- `/api/camera-config`
- `/api/camera-state`
- `/api/camera-frame?camera_id=...`
- `/api/camera-stream?camera_id=...`
- `/stream/camera/{camera_id}.mjpg`
- `/api/calibration/state`
- `/api/calibration/preview?camera_id=...`
- `/api/calibration/save`
- `/api/calibration/reset`
- `/artifact?path=...`

OpenAPI docs are in `docs/api/openapi.yaml`.

## Command

```cmd
cd /d "<repo-root>"
powershell -ExecutionPolicy Bypass -File ".\run_live_event_demo_server.ps1"
```

Default URL:

- `http://127.0.0.1:8765`

Direct server command:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_live_event_demo_server.py" --host 127.0.0.1 --port 8765 --project-root "." --output-root ".\outputs\evaluations\p0_repair_eval_a1\offline_runs\a1" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --stream-target-fps 15
```

## Run C1 Demo Data

The `c1` clip pair is a newly added New Dataset case. It is not a training run; it uses the existing pretrained detector/tracker/face models and produces runtime artifacts for the Live Demo UI.

Expected source files:

- `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\c1.mp4`
- `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 2\c1.mp4`

The generated run used by the UI is:

- `outputs/evaluations/c1_demo_artifacts/offline_runs/c1`

Run the server with Command Prompt:

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
run_live_event_demo_server_c1.cmd
```

Or run the explicit command:

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
"D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_live_event_demo_server.py" --project-root "." --dataset-root "%DATASET_ROOT%" --demo-pair-id c1 --output-root ".\outputs\evaluations\c1_demo_artifacts\offline_runs\c1" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --stream-target-fps 15 --presentation-mode sequential --camera-sequence C1,C2,C3,C4 --camera-segment-sec 12 --travel-gap-sec 3
```

Important: `--demo-pair-id c1` is required. Without it, camera streams would keep using the calibration preview source, which may point to `a1.mp4`, while event cards come from the `c1` output root.

Current `c1` limitation from the generated artifacts:

- CAM1 produced `0` raw detections in the configured 0-120 frame low-load window, so C1/C3 have no track rows or ENTRY_IN events.
- CAM2 produced `31` detections and emitted C2/C4 ENTRY_IN events.
- The UI can show `c1` streams and artifacts, but `c1` is currently a Camera-2-only event case under this smoke window.

FPS benchmark command:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_server_fps_benchmark.py" --project-root "." --output-root ".\outputs\evaluations\p0_repair_eval_a1\offline_runs\a1" --duration-sec 6 --target-fps 15 --summary-json ".\outputs\evaluations\server_fps_benchmark\server_runtime_summary.json"
```

The P0 validation run measured average worker FPS `14.885` across four streams.

## Current Limitation

This UI is still a thesis tool:

- it is not a production monitoring dashboard
- it depends on artifacts already written by the pipeline
- if backend association is wrong, the UI will expose that wrongness rather than hide it
