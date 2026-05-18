# Lightweight Live Demo UI

The UI in this repository is a thesis-debugging surface, not a separate product. It is used for:

1. manual ROI / calibration verification
2. event, timeline, and Re-ID evidence inspection
3. defense-day demo playback from offline artifacts

## Runtime Pieces

Server:

- `insightface_demo_assets/runtime/run_live_event_demo_server.py`
- `insightface_demo_assets/runtime/run_server_fps_benchmark.py`

Wrappers:

- `run_live_event_demo_server_b2.cmd`
- `run_live_event_demo_server_c1.cmd`

Static pages:

- `insightface_demo_assets/runtime/web_demo/index.html`
- `insightface_demo_assets/runtime/web_demo/calibration.html`

## New Dataset Demo Defaults

These defaults are specific to the current New Dataset thesis demo:

- YOLO person detector confidence threshold: `0.6`
- person class filter: `0`
- default association policy: `association_policy.new_dataset_demo.yaml`
- topology-supported dynamic thresholding: enabled by default for the New Dataset demo
- default Known DB root: `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID`

The `0.6` detector threshold was promoted from the `a3` audit because it reduced likely false positives without removing the validated ENTRY_IN events in that case. It is not documented here as a universal conclusion for all scenes.

## Frame Serving Model

The server can read both live-run and offline-run artifacts. In the current demo runtime it also starts background replay workers that decode and annotate frames independently of the HTTP request path.

Frame serving:

- background workers decode video and draw overlays
- each worker writes the latest JPEG and metadata into an in-memory buffer
- `/api/camera-frame` returns the latest buffered JPEG
- `/api/camera-stream` and `/stream/camera/{camera_id}.mjpg` stream MJPEG from the same buffer
- the browser uses MJPEG `<img>` streams for camera tiles and polls `/api/camera-state` only for metadata

This avoids the old bottleneck where `/api/camera-frame` performed video seeking and preview rendering per request.

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

## Run B2 Demo Data

First generate the `b2` runtime artifacts with the promoted detector threshold and default dynamic association policy:

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
"D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_b2_known_face_validation.py" --project-root "." --pair-id b2
```

Then start the UI:

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
run_live_event_demo_server_b2.cmd
```

Equivalent direct server command:

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
"D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_live_event_demo_server.py" --project-root "." --dataset-root "%DATASET_ROOT%" --demo-pair-id b2 --output-root ".\outputs\evaluations\b2_known_face_runtime_validation\offline_runs\b2" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --stream-target-fps 15 --presentation-mode sequential --camera-sequence C1,C2,C3,C4 --camera-segment-sec 12 --travel-gap-sec 3
```

## Run C1 Demo Data

Expected source files:

- `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\c1.mp4`
- `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 2\c1.mp4`

The generated run used by the UI is:

- `outputs/evaluations\c1_demo_artifacts\offline_runs\c1`

Command Prompt wrapper:

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
run_live_event_demo_server_c1.cmd
```

Explicit server command:

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
"D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_live_event_demo_server.py" --project-root "." --dataset-root "%DATASET_ROOT%" --demo-pair-id c1 --output-root ".\outputs\evaluations\c1_demo_artifacts\offline_runs\c1" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --stream-target-fps 15 --presentation-mode sequential --camera-sequence C1,C2,C3,C4 --camera-segment-sec 12 --travel-gap-sec 3
```

Important: `--demo-pair-id` is required. Without it, the camera streams may keep using the calibration preview source while the event/timeline cards come from a different output root.

## Endpoint Smoke Checks

After the server is up:

```cmd
curl http://127.0.0.1:8765/api/latest-events
curl http://127.0.0.1:8765/api/timeline
curl http://127.0.0.1:8765/api/reid-handoffs
curl http://127.0.0.1:8765/api/camera-config
curl "http://127.0.0.1:8765/api/camera-state?demo_time_sec=10"
```

Open the browser:

- `http://127.0.0.1:8765`
- `http://127.0.0.1:8765/calibration.html`

## FPS Benchmark Command

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_server_fps_benchmark.py" --project-root "." --output-root ".\outputs\evaluations\b2_known_face_runtime_validation\offline_runs\b2" --duration-sec 6 --target-fps 15 --summary-json ".\outputs\evaluations\server_fps_benchmark\server_runtime_summary.json"
```

The previous P0 validation run measured average worker FPS `14.885` across four streams.

## Current Limitation

This UI is still a thesis tool:

- it is not a production monitoring dashboard
- it depends on artifacts already written by the pipeline
- if backend association is wrong, the UI will expose that wrongness rather than hide it
