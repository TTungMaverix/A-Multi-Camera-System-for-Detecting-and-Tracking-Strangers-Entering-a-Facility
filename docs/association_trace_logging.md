# Association Trace Logging

The demo now writes decision logs to:

- `insightface_demo_assets/runtime/association_logs/association_decisions.jsonl`
- `insightface_demo_assets/runtime/association_logs/association_summary.json`
- `insightface_demo_assets/runtime/association_logs/association_policy_runtime.json`

These files are runtime outputs and are not intended to be committed.

## Decision Log Schema

Each JSONL row corresponds to one observation event and includes:

- `timestamp_sec`
- `relative_time`
- `camera_id`
- `observation_id`
- `event_type`
- `quality_gate_pass`
- `quality_gate_reason`
- `candidate_set_before_filter`
- `candidate_set_after_filter`
- `selected_candidate_id`
- `relation_type`
- `topology_metadata`
- `time_delta`
- `travel_window`
- `modality_primary`
- `modality_secondary`
- `face_score`
- `body_score`
- `thresholds_used`
- `margin_used`
- `decision`
- `reason_code`
- `gallery_id_before`
- `gallery_id_after`
- `candidate_evaluations`

## Summary Metrics

`association_summary.json` currently records:

- `decision_count`
- `known_accept_count`
- `unknown_reuse_count`
- `new_unknown_count`
- `defer_count`
- `quality_gate_reject_count`
- `topology_reject_count`

These metrics are also merged into `face_resolution_summary.json` for the mode-B run.

## Purpose

This trace layer is intended for:

- debugging why a reuse happened or failed
- explaining decisions during thesis demo and defense
- supporting later dashboard/storage phases without changing the association core again
