# Phase F Tuning Rerun

## Purpose

The previous selected policy was tuned on an older candidate-event distribution.

After Phase F changed:

- body descriptor quality
- body fallback logic
- `C5/C6` spatial assignment
- line-aware best-shot behavior

the policy had to be tuned again.

## Config

New sweep config:

- `insightface_demo_assets/runtime/config/association_tuning_grid.phase_f.yaml`

New selected policy output:

- `insightface_demo_assets/runtime/config/association_policy.wildtrack_phase_f_tuned.yaml`

## Evaluated Variants

- `phase_f_current_policy`
- `phase_f_overlap_body_relaxed`
- `phase_f_overlap_body_guarded`
- `phase_f_overlap_body_aggressive`

## Selected Variant

- `phase_f_overlap_body_aggressive`

## Selected Metrics

- `pairwise_f1 = 0.4498`
- `pairwise_precision = 0.6325`
- `pairwise_recall = 0.3491`
- `TRUE_MODEL_BASED_REUSE_COUNT = 27`
- `UNIQUE_UNKNOWN_IDS = 86`
- `REUSED_UNKNOWN_IDS = 45`
- `merge_error_count = 18`
- `split_gt_count = 41`

## Interpretation

This met the supervisor target band for the demo set:

- target: around `0.3 - 0.4` if achievable
- result: `0.4498`

The system is still not perfect:

- `merge_error_count` is not low enough to call the problem solved
- some wide-camera overlap cases still merge too aggressively

But the core thesis requirement is now substantially stronger:

- unknown-to-unknown cross-camera reuse is real
- body-first overlap matching is working
- the selected policy is no longer tuned on stale pre-fix events
