# Association Evaluation and Threshold Tuning

## Goal

The current tuning workflow avoids blind threshold edits.

It uses cached candidate events, cached crops, and repeated policy sweeps over the same event set so different settings can be compared on the same data.

For the current New Dataset phase, threshold tuning is intentionally **not** the first step. The workflow is:

1. validate direction logic independently
2. audit dataset coverage and calibration reuse
3. inspect face/body evidence quality and crop quality
4. benchmark pooling / extractor variants
5. only then consider threshold changes

This is why the current phase adds standalone validation scripts instead of immediately changing `body_primary`.

## Entry Point

Legacy policy sweep entry point:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_association_tuning.py" --config ".\insightface_demo_assets\runtime\config\association_tuning_grid.example.yaml"
```

## Inputs

- generated candidate events
- known face gallery + manifest
- base association policy
- camera transition map
- dataset inventory and calibration reuse audit for the New Dataset phase

## Parameters Considered

The workflow is meant to compare settings for:

- `known_accept_threshold`
- `unknown_reuse_threshold`
- `margin_by_relation`
- `relation_thresholds.overlap`
- `relation_thresholds.sequential`
- `relation_thresholds.weak_link`
- quality-gate thresholds if a variant overrides them
- `body_reid.extractor_name`
- `body_reid.tracklet_pooling_mode`

## Current New Dataset Validation Scripts

Independent direction validation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_direction_validation.py" --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" --output-root "outputs/evaluations/direction_validation_tracklet_phase"
```

Dataset inventory + calibration reuse audit:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_inventory.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --output-dir "outputs/evaluations/a2_a3_cv_phase_inventory"
```

Per-clip evaluation across all currently paired local clips:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_evaluation.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --inventory-json ".\outputs\evaluations\a2_a3_cv_phase_inventory\dataset_inventory.json" --calibration-reuse-json ".\outputs\evaluations\a2_a3_cv_phase_inventory\calibration_reuse_summary.json" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current" --baseline-output-dir ".\outputs\evaluations\new_dataset_quality_pooling_phase_current"
```

a2 overlay debug:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_pair_debug.py" --pipeline-config ".\outputs\evaluations\a2_a3_cv_phase_current\tmp_phase_pipeline.yaml" --run-output-root ".\outputs\evaluations\a2_a3_cv_phase_current\offline_runs\a2" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current\a2_debug" --pair-id a2
```

a3 traditional-CV benchmark:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_a3_hard_case_analysis.py" --run-output-root ".\outputs\evaluations\a2_a3_cv_phase_current\offline_runs\a3" --output-dir ".\outputs\evaluations\a2_a3_cv_phase_current\a3_hard_case" --pair-id a3 --association-policy-config ".\insightface_demo_assets\runtime\config\association_policy.new_dataset_demo.yaml"
```

The current per-clip evaluation writes:

- `per_clip_evaluation/<pair>.json`
- `overall_evaluation_summary.json`
- `appearance_vs_topology_summary.json`
- `face_branch_summary.json`
- `qualitative_case_notes.md`
- `regression_summary.json`

These scripts are the required evidence before arguing for any further threshold change.

## Current Comparison Metrics

- `appearance_only_body_score`
- `primary_threshold`
- `topology_support_level`
- `final_decision`
- `acceptance_reason`
- `appearance_only_pass_count`
- `topology_supported_pass_count`
- `topology_rescued_count`
- `unknown_reuse_count`
- `new_unknown_count`
- `face_candidate_count`
- `face_best_shot_selected_count`
- `face_embedding_created_count`

## Current Phase Note

The current New Dataset policy keeps:

- `relation_thresholds.sequential.body_primary = 0.72`

The repo no longer treats `0.55` as an acceptable default. If a sequential candidate is accepted slightly below `0.72`, that acceptance must be explained explicitly by the topology-supported sequential rule and appear in the decision logs with a concrete reason code.

Current local evaluation snapshot for `outputs/evaluations/a2_a3_cv_phase_current`:

- paired clips actually present locally: `a1`, `a2`, `a3`, `b1`
- appearance-only pass count: `4`
- topology-supported pass count: `2`
- topology rescue count: `2`
- average `osnet_mean` body score on compared clips: `0.586`
- average `osnet_quality_aware` body score on compared clips: `0.5856`
- average `osnet_x1_0_quality_aware` body score on compared clips: `0.6084`
- `a2` improved from `TOTAL_EVENTS = 0` to `TOTAL_EVENTS = 4`
- `a3` best appearance-only score improved from `0.5071` to `0.58` with traditional CV preprocessing and bbox shrink

Interpretation:

- topology rescue is still needed on the true physical `C1 -> C2` cross-view pair for `a1` and `b1`
- `a2` no longer fails at event creation; it now fails because appearance-only scores remain near `0.60`
- `a3` remains a real cross-view appearance failure even after shrink/preprocessing
- this phase therefore focuses on input quality and diagnostic evidence, not threshold rollback
