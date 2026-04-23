# Association Trace Logging

The demo now writes decision logs to:

- `insightface_demo_assets/runtime/association_logs/association_decisions.jsonl`
- `insightface_demo_assets/runtime/association_logs/association_summary.json`
- `insightface_demo_assets/runtime/association_logs/association_policy_runtime.json`
- `insightface_demo_assets/runtime/association_logs/camera_transition_map_runtime.json`

These files are runtime outputs and are not intended to be committed.

## Decision Log Schema

Each JSONL row corresponds to one observation event and includes:

- `timestamp_sec`
- `relative_time`
- `camera_id`
- `observation_id`
- `event_type`
- `zone_id`
- `zone_type`
- `subzone_id`
- `subzone_type`
- `quality_gate_pass`
- `quality_gate_reason`
- `candidate_set_before_filter`
- `candidate_set_after_filter`
- `selected_candidate_id`
- `relation_type`
- `transition_rule_used`
- `source_camera_id`
- `target_camera_id`
- `topology_metadata`
- `time_delta`
- `observed_delta_sec`
- `travel_window`
- `time_distance_to_expected_sec`
- `topology_support_level`
- `source_zone_id`
- `target_zone_id`
- `zone_valid`
- `zone_reason`
- `fallback_without_zone`
- `source_subzone_id`
- `target_subzone_id`
- `subzone_valid`
- `subzone_reason`
- `fallback_without_subzone`
- `modality_primary`
- `modality_secondary`
- `face_score`
- `body_score`
- `thresholds_used`
- `margin_used`
- `acceptance_reason`
- `decision`
- `reason_code`
- `gallery_id_before`
- `gallery_id_after`
- `candidate_evaluations`

For the current New Dataset phase, sequential body-only reuse can also log:

- `thresholds_used.topology_supported_accept = true`
- `thresholds_used.topology_supported_shortfall`

This makes it clear when a candidate was accepted because topology/time/zone/subzone support
was strong enough to rescue a near-threshold body score without lowering the global threshold.

## Summary Metrics

`association_summary.json` currently records:

- `decision_count`
- `known_accept_count`
- `unknown_reuse_count`
- `new_unknown_count`
- `defer_count`
- `quality_gate_reject_count`
- `topology_reject_count`
- `zone_reject_count`
- `fallback_without_zone_count`
- `subzone_reject_count`
- `fallback_without_subzone_count`

The current face/body usage summary exported by `run_face_resolution_demo.py` also records:

- `face_embedding_created_count`
- `face_reject_yaw_count`
- `face_reject_pitch_count`
- `face_reject_size_count`
- `face_reject_camera_disabled_count`
- `body_tracklet_candidate_crop_count`
- `body_tracklet_valid_crop_count`
- `body_tracklet_selected_crop_count`
- `body_primary_decision_count`
- `body_only_decision_count`

These metrics are also merged into `face_resolution_summary.json` for the mode-B run.

## Purpose

This trace layer is intended for:

- debugging why a reuse happened or failed
- explaining decisions during thesis demo and defense
- supporting later dashboard/storage phases without changing the association core again
- preserving enough zone-aware metadata for a later timeline/dashboard layer
- explaining whether a failure came from geometry, missing face evidence, or weak body evidence
