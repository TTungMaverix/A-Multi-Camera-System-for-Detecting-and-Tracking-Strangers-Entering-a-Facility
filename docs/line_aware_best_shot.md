# Line-Aware Best-Shot Selection

## Goal

The offline pipeline should not lock onto a crop too early when a person has only just crossed an entry line.

The current best-shot logic therefore prefers frames that:

- happen after the entry or follow-up anchor
- land in a more useful subzone such as `exit` or `interior`
- still keep a reasonably large body box for crop quality

For the current New Dataset phase, line-aware best-shot is now split by modality intent:

- body/event best shots can still come from high-angle anchor cameras
- face best shots are only attempted on cameras whose `face_capture_mode` allows face capture
- face candidates must also pass bbox-size and pose gates before an embedding is created

## Config

The current dataset-facing config lives in:

- [dataset_profile.new_dataset_demo.yaml](../insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml)
- [association_policy.new_dataset_demo.yaml](../insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml)

Relevant keys:

- `best_shot_selection.enabled`
- `best_shot_selection.preferred_subzone_types`
- `best_shot_selection.minimum_frames_after_anchor`
- `cameras[].face_capture_mode`
- `association_policy.quality_gate.min_face_bbox_width`
- `association_policy.quality_gate.min_face_bbox_height`
- `association_policy.quality_gate.min_face_bbox_area`
- `association_policy.quality_gate.max_abs_yaw_deg`
- `association_policy.quality_gate.max_abs_pitch_deg`

## Runtime Behavior

When line-aware mode is enabled, best-shot ranking is:

1. frames that satisfy the minimum post-anchor delay
2. preferred subzone type priority
3. larger bbox area
4. larger post-anchor frame distance

If line-aware mode is disabled or spatial context is missing, the pipeline falls back to area-based selection inside the configured frame window.

## Audit Fields

The event-generation audit now records:

- `best_shot_strategy`
- `best_shot_reason`
- `best_shot_zone_id`
- `best_shot_subzone_id`
- `best_shot_subzone_type`
- `best_shot_frames_after_anchor`

The face/body audit now also records:

- `face_candidate_count`
- `face_best_shot_selected_count`
- `face_embedding_created_count`

This makes it possible to inspect why a given frame was selected as the event crop and whether a face best shot really produced usable evidence.

## Current Limitation

On the current local New Dataset evaluation sweep (`a1`, `a2`, `a3`, `b1`):

- `C1` / `C3` are intentionally body-anchor views, not face sources
- most face candidates are still rejected by camera role or yaw before embedding
- `b1` is the only current local clip that produced:
  - `face_best_shot_selected_count = 1`
  - `face_embedding_created_count = 1`
- even on `b1`, that face evidence did not yet become a decisive cross-camera anchor because there was no strong face gallery reference for the reuse step

That is still an improvement over silent failure: the current phase now makes best-shot selection and rejection explicit, so the remaining blocker is visible as evidence quality and availability, not hidden behavior.
