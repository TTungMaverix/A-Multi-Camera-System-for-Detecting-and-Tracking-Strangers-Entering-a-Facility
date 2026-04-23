# Association Evaluation and Threshold Tuning

## Goal

The current tuning workflow avoids blind threshold edits.

It uses cached candidate events, cached crops, and repeated policy sweeps over the same event set so different settings can be compared on the same data.

For the current New Dataset phase, threshold tuning is intentionally **not** the first step.
The workflow is:

1. validate direction logic independently
2. audit face/body evidence quality and crop quality
3. fix map / topology / subzone issues
4. only then consider threshold changes

This is why the current phase adds standalone validation scripts instead of immediately
changing `body_primary`.

## Entry Point

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_association_tuning.py" --config ".\insightface_demo_assets\runtime\config\association_tuning_grid.example.yaml"
```

## Inputs

- `generated_candidate_events_mode_b.csv`
- known face gallery + manifest
- base association policy
- camera transition map

## Parameters Considered

The workflow is meant to compare settings for:

- `known_accept_threshold`
- `unknown_reuse_threshold`
- `margin_by_relation`
- `relation_thresholds.overlap`
- `relation_thresholds.sequential`
- `relation_thresholds.weak_link`
- quality-gate thresholds if a variant overrides them

## Output Files

Under `outputs/evaluation/association_policy_sweep/`:

- `association_policy_sweep.csv`
- `association_policy_sweep.json`
- `candidate_event_feature_cache.json`
- `known_face_embeddings_eval.csv`
- `selected_policy_summary.json`
- `selected_policy_summary.md`

And the selected policy is written to:

- `insightface_demo_assets/runtime/config/association_policy.wildtrack_tuned.yaml`

Important note:

- the current tuned policy was selected before the latest line-aware best-shot refinement
- if best-shot behavior changes materially, rerun the policy sweep so thresholds are chosen against the new candidate-event distribution

## Current New Dataset Validation Scripts

Independent direction validation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_direction_validation.py" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --output-root "outputs/evaluations/direction_validation_tracklet_phase"
```

Tracklet-body comparison on a concrete offline run:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_body_tracklet_evaluation.py" --run-output-root "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4" --output-dir "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet"
```

These two scripts are the required evidence before arguing for any further threshold change.

## Current Comparison Metrics

- `known_accept_count`
- `unknown_reuse_count`
- `new_unknown_count`
- `defer_count`
- `unique_unknown_id_count`
- `reused_unknown_id_count`
- `true_model_based_reuse_count`
- pairwise association `precision / recall / f1`
- `merge_error_count`
- `split_gt_count`

## Selection Rule

The current implementation ranks variants by:

1. highest `pairwise_f1`
2. highest `true_model_based_reuse_count`
3. lowest `merge_error_count`
4. lowest `split_gt_count`

This keeps the workflow interpretable and paper-grounded without adding a heavy new model.

## Current Phase Note

The current New Dataset policy keeps:

- `relation_thresholds.sequential.body_primary = 0.72`

The repo no longer treats `0.55` as an acceptable default. If a sequential candidate is
accepted slightly below `0.72`, that acceptance must be explained explicitly by the
topology-supported sequential rule and appear in the decision logs with a concrete reason code.
