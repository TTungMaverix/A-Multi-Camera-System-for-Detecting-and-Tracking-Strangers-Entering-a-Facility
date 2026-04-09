# Line-Aware Best-Shot Selection

## Goal

The offline pipeline should not lock onto a crop too early when a person has only just crossed an entry line.

The current best-shot logic therefore prefers frames that:

- happen after the entry or follow-up anchor
- land in a more useful subzone such as `exit` or `interior`
- still keep a reasonably large body box for crop quality

## Config

The current dataset-facing config lives in:

- [wildtrack_demo_config.json](../wildtrack_demo/wildtrack_demo_config.json)

Relevant keys:

- `best_shot_selection.enabled`
- `best_shot_selection.preferred_subzone_types`
- `best_shot_selection.minimum_frames_after_anchor`

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

This makes it possible to inspect why a given frame was selected as the event crop.

## Current Limitation

The current Wildtrack subzone map is still approximate. Because of that, line-aware selection can expose map-definition problems:

- some events still land in `outer_entry`
- stricter subzone-aware filtering can reduce reuse if the chosen subzones do not reflect the true movement path well

That limitation is expected at the current stage and is exactly why the new audit fields are exported.
