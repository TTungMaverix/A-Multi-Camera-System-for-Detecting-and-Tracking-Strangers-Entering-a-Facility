# Camera Transition Map Config

Map-aware association metadata is externalized at:

- [camera_transition_map.example.yaml](../insightface_demo_assets/runtime/config/camera_transition_map.example.yaml)

This file is the dataset-facing hook for:

- per-camera zone definitions
- entry / exit map hints
- directed camera-pair transitions
- overlap vs sequential vs weak-link behavior
- expected travel-time windows

## Loader Behavior

The runtime checks:

1. `camera_transition_map_config` from `face_demo_config.json`
2. `runtime/config/camera_transition_map.yaml`
3. `runtime/config/camera_transition_map.example.yaml`
4. fallback metadata derived from `wildtrack_demo_config.json`

Missing keys are filled from the fallback map so the vertical slice keeps running.

## Schema Overview

Top-level sections:

- `cameras`
- `transitions`

Each `camera` can declare:

- `role`
- `description`
- `default_zone_id`
- `entry_zones`
- `exit_zones`
- `zones`

Each zone supports:

- `zone_id`
- `zone_type`
- `polygon`
- `placeholder`
- `description`

Each `transition` supports:

- `transition_id`
- `src_camera_id`
- `dst_camera_id`
- `relation_type`
- `allowed_relation_types`
- `same_area_overlap`
- `min_travel_time_sec`
- `avg_travel_time_sec`
- `max_travel_time_sec`
- `allowed_exit_zones`
- `allowed_entry_zones`
- `weak_link_support`
- `description`

## Runtime Use

During event construction:

- observations are assigned `zone_id` / `zone_type` from the configured zone polygons when possible
- if no polygon matches, the runtime falls back to the camera default zone
- the event keeps `zone_reason` and `zone_fallback_used` for audit

During association:

- topology filter checks camera-pair transition validity
- time filter checks the transition travel window
- zone filter checks source-zone to target-zone compatibility
- logs record whether a candidate was rejected by topology, time, or zone

## Dataset Adaptation

For a new dataset:

1. copy the example file
2. replace camera ids and zone polygons
3. define directed transitions with realistic travel-time windows
4. keep the association core code unchanged

This keeps the implementation consistent with the paper-grounded design note while staying interpretable and debug-friendly.
