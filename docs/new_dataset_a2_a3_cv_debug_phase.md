# New Dataset A2/A3 CV Debug Phase

## Scope

This phase stays within the current offline vertical slice and does not add a new product surface, new deployment mode, or new ReID model family.

The phase only does three things:

1. debug why `a2` produced `TOTAL_EVENTS = 0`
2. treat the `a3` hard case with traditional CV preprocessing
3. rerun `a1`, `a2`, `a3`, `b1` and report regressions honestly

## Dataset Coverage

Current local paired clips:

- `a1`
- `a2`
- `a3`
- `b1`

Inventory artifact:

- `outputs/evaluations/a2_a3_cv_phase_inventory/dataset_inventory.json`

Calibration reuse artifact:

- `outputs/evaluations/a2_a3_cv_phase_inventory/calibration_reuse_summary.json`

All four clip pairs reuse the same source-camera calibration successfully. No per-clip redraw was needed in this phase.

## A2 Root Cause

Before this phase, `a2` failed because tracks started late and already inside the protected side of the entry line, so the original line-crossing-only logic never emitted an `ENTRY_IN` event.

The fix was not a threshold hack. The fix was a constrained late-start inside-entry fallback in the direction/event stage.

Current evidence:

- overlay: `outputs/evaluations/a2_a3_cv_phase_current/a2_debug/a2_overlay_debug.mp4`
- summary: `outputs/evaluations/a2_a3_cv_phase_current/a2_debug/a2_stage_debug_summary.json`
- report: `outputs/evaluations/a2_a3_cv_phase_current/a2_debug/a2_root_cause_report.md`

Current result:

- `a2 TOTAL_EVENTS = 4`
- detector alive
- tracker alive
- event creation alive
- cross-camera reuse still fails because the recovered appearance-only body scores stay around `0.6001 .. 0.6007`

## A3 Hard Case

`a3` remains the strongest appearance failure case in the local dataset.

Visual issues confirmed by crop dump and contact sheet:

- strong background clutter
- scale variance between cameras
- body crops include too much background
- color/illumination shift between the two source cameras

Artifacts:

- crop dump:
  - `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_crops/c1/`
  - `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_crops/c2/`
- contact sheet:
  - `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_contact_sheet.png`
- preprocessing benchmark:
  - `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_preprocessing_benchmark.json`
- bbox shrink benchmark:
  - `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_bbox_shrink_benchmark.json`
- report:
  - `outputs/evaluations/a2_a3_cv_phase_current/a3_hard_case/a3_hard_case_report.md`

Best practical combination in this phase:

- `gray_world` preprocessing
- `bbox_shrink_ratio = 0.1`

Observed improvement:

- baseline hard-case score: `0.5071`
- best current score after CV treatment: `0.58`

That is a real gain, but it is still below the global sequential body threshold of `0.72`.

## Cross-Clip Outcome

Evaluation output:

- `outputs/evaluations/a2_a3_cv_phase_current/overall_evaluation_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/regression_summary.json`
- `outputs/evaluations/a2_a3_cv_phase_current/per_clip_evaluation/*.json`

Clip-level outcome:

- `a1`: still reuses across cameras; no regression on the main chain
- `a2`: fixed from `0` events to `4` events; reuse still fails
- `a3`: still fails cross-camera reuse; appearance improves but not enough
- `b1`: still reuses across cameras; no regression on the main chain

## Remaining Blockers

- the dataset still has only `4` paired clips, not the target `5`
- `a2` now fails at appearance, not event creation
- `a3` still does not clear appearance-only acceptance
- topology/time remains necessary on the true physical `C1 -> C2` pair
- face evidence is still sparse and not yet a decisive identity anchor on most clips
