# Phase F C5/C6 Map Fix

## Files Changed

- `insightface_demo_assets/runtime/config/camera_transition_map.example.yaml`
- `wildtrack_demo/wildtrack_demo_config.json`

## Why C5/C6 Were Prioritized

The audit showed that the biggest entry-camera fragmentation came from `C6`.

Many overlap observations were still landing in `c6_outer_entry`, even when they were already usable for cross-camera association.

That created artificial rejects against:

- `C5 inner_exit`
- `C3 overlap_entry`
- `C7 overlap_entry / far_exit`

## Main Fixes

### 1. Moved the `C6` outer/inner boundary upward

`c6_inner_exit` was moved upward to better align with the real line-crossing region.

Effect:

- the entry-camera best-shot audit changed from many `c6_outer_entry` best shots to mostly `c6_inner_exit`
- current entry-event result:
  - `C6 inner_exit = 12`
  - `C6 outer_entry = 1`

### 2. Allowed `c6_outer_entry` for overlap transitions

For overlap relations, a stranger may still be visible in the outer approach band while already being visible in a neighboring camera.

So the transition rules now allow `c6_outer_entry` in `C6` overlap transitions instead of forcing `c6_inner_exit` only.

### 3. Broadened `C7` overlap entry support

`C7` is a wide follow-up view, so overlap observations can land in either:

- `c7_overlap_entry`
- `c7_far_exit`

The overlap transition rules were updated accordingly.

### 4. Increased line-aware post-anchor delay for overlap cases

`wildtrack_demo_config.json` now requires more frames after the anchor for overlap/sequential/weak-link best-shot selection.

This reduces early captures that stay in poor subzones.

## Visual Evidence

Overlay artifacts committed for review:

- [docs/assets/phase_f/C5_subzones_overlay.png](assets/phase_f/C5_subzones_overlay.png)
- [docs/assets/phase_f/C6_subzones_overlay.png](assets/phase_f/C6_subzones_overlay.png)

## Result

The map fix removed most of the earlier `C6` entry-event placement problem and made the remaining errors much more clearly about appearance thresholds instead of bad spatial assignment.
