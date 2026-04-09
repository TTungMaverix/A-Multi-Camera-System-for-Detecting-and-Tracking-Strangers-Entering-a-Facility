# Offline Multiprocessing Architecture

## Goal

The current multiprocessing baseline keeps the thesis architecture simple:

- 4 edge workers
- 1 central brain
- offline video files first
- no external broker
- no per-worker global ID assignment

## Edge Workers

Each worker owns exactly one camera.

Worker responsibilities:

1. read the configured source for that camera
2. build GT-backed local track rows from Wildtrack annotations
3. apply local direction filtering for entry cameras
4. emit lightweight packets to a `multiprocessing.Queue`

Workers do **not**:

- run known/unknown final resolution
- manage the unknown gallery
- assign `Unknown_Global_ID`
- run cross-camera association

## Packet Types

Current packet types:

- `track_row_batch`
- `entry_anchor_batch`
- `camera_summary`
- `worker_started`
- `worker_done`
- `worker_error`

Entry-anchor packets intentionally stay lightweight:

- `camera_id`
- `frame_idx`
- `relative_sec`
- `local_track_id`
- `global_gt_id` for audit only
- `direction`
- `bbox_*`
- `foot_x`
- `foot_y`
- `zone_id`
- `subzone_id`
- `crop_reference`

No raw frame arrays are sent through the queue.

## Central Brain

The main process receives packets, aggregates local observations, then performs:

1. stage-input materialization
2. face embedding / known DB matching
3. unknown gallery update
4. cross-camera association
5. final event, timeline, audit, and summary exports

This keeps Global Unknown ID management centralized.

## Logging and Shutdown

Each multiprocessing run writes:

- `outputs/offline_runs/<run_name>/logs/multiprocessing_events.json`
- `outputs/offline_runs/<run_name>/logs/multiprocessing_summary.json`
- `outputs/offline_runs/<run_name>/logs/edge_entry_anchor_packets.csv`

Workers terminate through an explicit `worker_done` packet. Errors are surfaced through `worker_error`.

## Modes

Use the same offline orchestrator with config:

- `execution.mode: sequential`
- `execution.mode: multiprocessing`

Sequential mode remains the debugging fallback.
