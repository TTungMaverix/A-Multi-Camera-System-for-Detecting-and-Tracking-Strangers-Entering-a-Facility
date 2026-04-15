# Association Evaluation and Threshold Tuning

## Goal

The current tuning workflow avoids blind threshold edits.

It uses cached candidate events, cached crops, and repeated policy sweeps over the same event set so different settings can be compared on the same data.

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
