# Live Pipeline Baseline

Phase G adds a lightweight live ingestion path on top of the existing offline thesis pipeline.

This phase is intentionally minimal:

- `4 Edge Workers` read live-capable sources independently
- each worker runs lightweight person detection + simple tracking
- workers emit only lightweight event packets to the central process
- the central process remains the only owner of:
  - face matching
  - known vs unknown resolution
  - unknown gallery
  - `Unknown_Global_ID`
  - cross-camera association

## Current Scope

Supported source types:

- `file`
- `rtsp`
- `webcam`

Current verified run in this repository:

- `file`-mode sanity replay through the live code path
- config: `insightface_demo_assets/runtime/config/live_pipeline_demo.file_sanity.yaml`

This is a pragmatic sanity step because the current Codex environment does not guarantee 4 real cameras or RTSP streams at the same time.

## Architecture

Runtime entrypoint:

- `insightface_demo_assets/runtime/run_live_multicam_demo.py`

PowerShell wrapper:

- `run_live_multicam_demo.ps1`

Core module:

- `insightface_demo_assets/runtime/live_pipeline/orchestrator.py`

Worker responsibilities:

- open `file / rtsp / webcam` source
- keep only the latest frame when configured
- load the manual scene calibration for that camera
- detect people with OpenCV HOG
- track with a lightweight centroid tracker
- apply ROI masking/filtering and trajectory-aware direction logic
- emit only event packets, not full frames

Central process responsibilities:

- load known face gallery once
- analyze event crops with InsightFace
- apply map-aware association
- update unknown gallery and IDs
- write live event stream outputs

## Latest-Frame / Frame-Drop Logic

Each worker uses `LatestFrameReader`.

Important config fields:

- `live.latest_frame_only`
- `live.target_fps`
- `live.reconnect_sleep_sec`
- `live.max_reconnect_attempts`
- `live.queue_put_timeout_sec`

Why this exists:

- without frame dropping, a live pipeline falls behind and starts processing stale frames
- for the thesis demo, a small amount of frame dropping is acceptable; very large latency is not

## Output Files

For a live run, outputs go to:

- `outputs/live_runs/<run_name>/events/live_event_stream.jsonl`
- `outputs/live_runs/<run_name>/events/latest_events.json`
- `outputs/live_runs/<run_name>/events/resolved_live_events.jsonl`
- `outputs/live_runs/<run_name>/association_logs/live_decision_stream.jsonl`
- `outputs/live_runs/<run_name>/association_logs/latest_decision_log.json`
- `outputs/live_runs/<run_name>/summaries/live_pipeline_summary.json`
- `outputs/live_runs/<run_name>/preview/<camera_id>_overlay.png`

## Manual Calibration Requirement

Live runtime now depends on:

- `insightface_demo_assets/runtime/config/manual_scene_calibration.wildtrack.json`

Missing or invalid manual calibration is a runtime error.

The old auto/inferred ROI fallback is no longer used in the live path.

## Commands

Default file-sanity run:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
powershell -ExecutionPolicy Bypass -File ".\run_live_multicam_demo.ps1"
```

Direct Python run:

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_live_multicam_demo.py" --config ".\insightface_demo_assets\runtime\config\live_pipeline_demo.file_sanity.yaml"
```

RTSP config template:

- `insightface_demo_assets/runtime/config/live_pipeline_demo.rtsp.example.yaml`

Webcam config template:

- `insightface_demo_assets/runtime/config/live_pipeline_demo.webcam.example.yaml`

## Current Limitations

- detection/tracking is intentionally lightweight, not the final production stack
- file-mode sanity replay still shows notable latency because central association is recomputed incrementally for simplicity
- RTSP/webcam support exists in code and config, but this repository has only been sanity-tested locally through file replay in the current environment
