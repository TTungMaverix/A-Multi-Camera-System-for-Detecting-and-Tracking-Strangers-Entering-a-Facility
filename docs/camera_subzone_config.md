# Camera Subzone Config

Subzones are the next layer below camera-level zones.

They let the association core reason about:

- which side of a camera view a stranger came from
- which interior band or overlap band they were seen in
- whether a transition is plausible for a specific entry/exit path

## Where Subzones Live

Subzones are currently stored inside:

- [camera_transition_map.example.yaml](../insightface_demo_assets/runtime/config/camera_transition_map.example.yaml)

This keeps camera zones, subzones, and directed transition rules in one dataset-facing file.

## Camera-Level Fields

Each camera can define:

- `default_subzone_id`
- `subzones`

Each subzone can define:

- `subzone_id`
- `parent_zone_id`
- `subzone_type`
- `polygon`
- `priority`
- `allowed_transitions`
- `placeholder`
- `description`

`priority` is used when more than one subzone polygon matches the same point.

## Transition-Level Fields

Each directed transition can define:

- `allowed_exit_subzones`
- `allowed_entry_subzones`

These are optional. If they are omitted, the filter falls back to zone-level logic.

## Runtime Behavior

During event generation:

1. use the current event foot point when available
2. assign `zone_id` from zone polygons
3. assign `subzone_id` from subzone polygons under that zone when possible
4. if no subzone matches, fall back to the camera default subzone
5. record audit fields:
   - `subzone_id`
   - `subzone_type`
   - `subzone_reason`
   - `subzone_fallback_used`

During association:

1. topology must allow the camera pair
2. time must fit the relation window
3. zone must be compatible if zone constraints exist
4. subzone must be compatible if subzone constraints exist
5. if subzone data is missing, the filter can fall back according to policy

## Current Wildtrack Example

The current example config gives each camera:

- a default zone
- a default subzone
- at least one approximate entry/overlap subzone
- at least one approximate exit/interior subzone

Some follow-up camera subzones are marked as placeholders because Wildtrack is not a clean facility-entry dataset. This is intentional and keeps the design honest while still making the map-aware logic testable.
