# Phase F Quality Diagnostic

## Scope

This diagnostic focuses on the supervisor's priority:

- unknown-to-unknown cross-camera association
- overlap / simultaneous visibility
- fragmentation across cameras
- why face-only reasoning is insufficient on the current dataset

## Pre-Fix Root Causes

Before the Phase F hotfixes, the strongest failure modes were:

1. `face` was unavailable for most events.
   - On the previous full run, face extraction failed on most Wildtrack events.
   - The dataset is multi-camera surveillance data, not a face-centric dataset.
   - As a result, true open-set reuse depended on body evidence, not known-face matching.

2. `C6` overlap events were often assigned to `c6_outer_entry`.
   - This pushed valid overlap observations into a subzone that the transition rules did not accept.
   - The main failing patterns were:
     - `C6 outer_entry -> C5 inner_exit`
     - `C6 outer_entry -> C3 overlap_entry`
     - `C6 outer_entry -> C7 far_exit / overlap_entry`

3. Body-only matching was structurally enabled, but too weak in practice.
   - The previous body descriptor was dominated by coarse HSV histograms.
   - Same-person and different-person body similarities were too close.
   - This produced many `below_primary_threshold` create-new decisions instead of reuse.

4. Overlap-heavy cases were the real bottleneck, not unknown count itself.
   - The dataset regularly shows the same person in multiple cameras at the same time or with very short delay.
   - The largest fragmentation clusters were:
     - `C3 <-> C5`
     - `C3 <-> C6`
     - `C5 <-> C6`
     - `C5 <-> C7`
     - `C3 <-> C7`

## Full-Run Hotfix Effect Before Re-Tuning

After the Phase F code and map hotfixes, but before the new policy sweep:

- `UNIQUE_UNKNOWN_IDS`: `155 -> 125`
- `REUSED_UNKNOWN_IDS`: `8 -> 30`
- `TRUE_MODEL_BASED_REUSE_COUNT`: `6 -> 11`
- `pairwise_f1`: approximately `0.1473` with the previous selected policy on the new event distribution

This confirmed that:

- the map/subzone fixes were helping
- body fallback was now actually carrying the association flow
- the remaining problem was mostly threshold/policy alignment on the new distribution

## Body Fallback Findings

On the post-fix full run:

- `body_fallback_used_count = 156`
- `face_unusable_event_count = 156`
- dominant face unusable reason:
  - `face_embedding_missing = 137`

This validates the supervisor's point:

- the system must not depend on face alone
- unknown association on this dataset is primarily a body-assisted, map-aware problem

## Simultaneous / Overlap Findings

The strongest unresolved overlap cases before tuning were still centered on:

- `C5 inner_exit <-> C3 overlap_entry`
- `C6 outer_entry <-> C5 inner_exit`
- `C6 outer_entry <-> C3 overlap_entry`
- `C5 inner_exit <-> C7 far_exit`
- `C3 overlap_entry <-> C7 far_exit`

These are not long travel-time failures.

They are mostly:

- same-area or near-simultaneous visibility
- body-score-threshold failures
- subzone mismatch from coarse overlap placement

## Conclusion

Phase F had to target exactly these issues:

1. improve body evidence quality
2. stop treating weak face as mandatory secondary evidence
3. fix `C5/C6` subzone placement for overlap cases
4. retune overlap thresholds on the regenerated candidate-event distribution
