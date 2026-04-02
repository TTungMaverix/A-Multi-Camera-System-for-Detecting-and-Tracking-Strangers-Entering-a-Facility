# Codex implementation brief - multi-camera stranger demo

## Goal

Build a runnable thesis demo for a four-camera security system that:
1. Tracks only people moving **into** the protected facility.
2. Extracts face embeddings with **InsightFace**.
3. Matches against a **known identity database**.
4. If no known match is found, creates or reuses a **global stranger ID**.
5. Stores event logs with `timestamp`, `camera_id`, `track_id`, `global_id`, `identity_type`, `snapshot_path`.
6. Uses a shared **camera topology map** and **travel-time windows** to predict which camera a stranger may appear in next.
7. Supports the real-world case where the same person may appear in multiple nearby cameras at the same time.

The implementation should stay close to the actual thesis flow already prepared in the local demo assets:
- 4 selected streams
- direction filtering first
- face check only after `IN`
- `known` or `unknown_global_id`
- shared topology and overlap-aware cross-camera association
- file-based outputs first
- dashboard/preview as optional presentation layer

## Environment assumptions for this machine

Use the environment that has already worked locally:
- prefer Python at `C:\Users\Admin\.platformio\python3\python.exe`
- create venv as `.venv_insightface_demo`
- use `onnxruntime` CPU by default
- reuse model cache root `C:\Users\Admin\.insightface`
- do not rely on `pip install insightface==0.7.3` directly from PyPI on Windows
- instead copy local source from `C:\Users\Admin\insightface-master\python-package\insightface`
- patch out optional imports that trigger unnecessary native build requirements

## Dataset-aligned demo profile

The current default example is aligned with the local Wildtrack-style demo split:
- entry cameras: `cam05`, `cam06`
- follow cameras: `cam03`, `cam07`

These names are examples only. The implementation must still support any 4 streams configured in YAML.

## Demo-first scope

Do not train face models from scratch.
Use pretrained models and embeddings.
Primary embedding model: **InsightFace buffalo_l**.

## Recommended stack

- Python 3.11
- FastAPI for backend + simple REST endpoints
- SQLite for demo database
- OpenCV for video I/O and drawing
- Ultralytics YOLO for person detection
- ByteTrack or BoT-SORT for per-camera tracking
- InsightFace for face detection + face embedding
- ONNXRuntime CPU by default; allow optional GPU mode if available
- Streamlit or FastAPI + simple HTML for dashboard

## Architecture

Implement these modules:
- `src/config/`
- `src/core/stream_manager.py`
- `src/core/person_pipeline.py`
- `src/core/direction_filter.py`
- `src/core/face_service.py`
- `src/core/known_registry.py`
- `src/core/unknown_registry.py`
- `src/core/topology.py`
- `src/core/association.py`
- `src/core/event_logger.py`
- `src/api/app.py`
- `src/ui/dashboard.py`
- `src/main.py`

## Functional requirements

### A. Input sources

Support both:
- video files
- RTSP streams

Each frame must carry:
- `camera_id`
- `timestamp`
- `frame_index`
- `image`

### B. Entering-direction filtering

Per camera, support one of these modes:
- line crossing
- ROI transition
- centroid vector rule

Only tracks classified as `IN` should continue to the face and identity pipeline.

### C. Known identity matching

- Build a known DB from folders of registered people.
- For each identity, compute 1..N InsightFace embeddings.
- Matching method: cosine similarity.
- If best similarity is above `known_match_threshold`, return `Known_ID`.
- Else route to unknown/stranger pipeline.

### D. Stranger profile creation

When a new stranger appears:
- create `unknown_global_id`
- store first camera, first time, last camera, last time
- store top-K best face frames and optional body crops
- store representative face embedding and optional body embedding
- store appearance history
- compute candidate next cameras from topology map

### E. Cross-camera association

When an unknown appears at camera `Cj` at time `t_now`:
1. Use the topology map to filter candidate strangers that can reach `Cj`.
2. Allow `min_travel_sec = 0` for overlapping views.
3. Compare the new unknown with candidate profiles using:
   - face similarity
   - optional body similarity
   - reference frame similarity against stored best shots
   - time score
   - topology score
4. Reuse existing `unknown_global_id` when final score exceeds threshold.
5. Otherwise create a new `unknown_global_id`.

### F. Same person visible in multiple cameras at the same time

This is required.
Because the selected demo area is compact, two nearby cameras may see the same person simultaneously.
So the association logic must support edges such as:
- `min_travel_sec = 0`
- `allow_overlap = true`
- no hard penalty for same-timestamp sightings in adjacent cameras with overlapping fields of view

### G. Prediction of next camera

For each stranger profile, compute a ranked list of next possible cameras from the topology graph.
Expose this in logs/API as:
- `predicted_next_cameras`
- each item with `{camera_id, min_sec, avg_sec, max_sec, confidence}`

### H. Logging and storage

Store at least:
- events table
- stranger_profiles table
- stranger_appearances table
- known_identities table

Required event fields:
- `event_id`
- `timestamp`
- `camera_id`
- `local_track_id`
- `global_id`
- `identity_type` (`known` or `unknown`)
- `direction`
- `known_match_score`
- `association_score`
- `snapshot_path`
- `predicted_next_cameras_json`

### I. Visualization and dashboard

The presentation layer is optional but strongly recommended for thesis demo quality.
Minimum supported views:
1. Live or near-live event table.
2. Filter by camera, time, global ID, identity type.
3. Stranger detail page with timeline across cameras.
4. Preview of best reference frames.
5. Next camera predictions.

Stretch presentation goal:
- show a 2x2 quad-view wall of the 4 cameras
- overlay `camera_id`, `track_id`, `global_id`, `known/unknown`, `direction`
- optionally stream the same information through a small web UI

Primary grading output still remains:
- captured stranger images
- time of appearance
- camera of appearance
- unified stranger history across streams

## Output contract for the demo

The demo is considered successful if it can show:
1. Person enters the facility in a camera.
2. The system keeps tracking the person.
3. Face is matched against known DB.
4. If unmatched, the system creates a stranger ID.
5. The event log shows camera, time, ID, snapshot.
6. When the same stranger appears in another camera, the system reuses the same global ID.
7. The system displays predicted next camera(s) from topology and travel-time constraints.
8. Optional preview or dashboard reflects the same IDs shown in the log outputs.

## Required repo layout

```text
project_root/
  config/
    app.yaml
    cameras.yaml
    topology.yaml
  data/
    streams/
      cam03.mp4
      cam05.mp4
      cam06.mp4
      cam07.mp4
    known_db/
      person_001/
      person_002/
    outputs/
      events/
      snapshots/
      clips/
      logs/
      debug/
    cache/
    db/
    models/
  docs/
    CODEX_IMPLEMENTATION_BRIEF.md
    CODEX_PROMPT.txt
  scripts/
    bootstrap_windows.ps1
    bootstrap_unix.sh
    build_known_db.py
    run_demo.py
  src/
    ...
  tests/
  vendor/
  requirements-demo.txt
  README.md
```

## Suggested scoring rule for unknown association

A simple rule-based score is enough:

`final_score = 0.55 * face_score + 0.15 * body_score + 0.15 * time_score + 0.15 * topology_score`

Fallback rule:
- if no reliable face is available, increase body/time/topology weights
- if overlapping cameras are configured, do not reject simultaneous sightings

## Acceptance tests Codex should implement

1. Build known DB embeddings from folder images.
2. Read 4 sources concurrently.
3. Filter out `OUT` tracks.
4. Create event logs only for `IN` tracks.
5. Assign `Known_ID` when similarity exceeds threshold.
6. Assign `Unknown_Global_ID` otherwise.
7. Reuse `Unknown_Global_ID` across cameras when association score passes threshold.
8. Support simultaneous visibility in overlapping cameras.
9. Expose event history and stranger details through dashboard or API.
10. Save snapshots for every event.
