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
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_inventory.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --output-dir "outputs/evaluations/new_dataset_inventory_phase_current"
```

Per-clip evaluation across all currently paired local clips:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_new_dataset_evaluation.py" --pipeline-config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml" --inventory-json ".\outputs\evaluations\new_dataset_inventory_phase_current\dataset_inventory.json" --calibration-reuse-json ".\outputs\evaluations\new_dataset_inventory_phase_current\calibration_reuse_summary.json" --output-dir ".\outputs\evaluations\new_dataset_quality_pooling_phase_current"
```

Tracklet-body comparison on a concrete offline run:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_body_tracklet_evaluation.py" --run-output-root "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4" --output-dir "outputs/offline_runs/new_dataset_logical_4cam_demo_tracklet_phase_smoke_v4/evaluation/body_tracklet_phase_current"
```

The per-clip evaluation writes:

- `per_clip_evaluation/<pair>.json`
- `overall_evaluation_summary.json`
- `appearance_vs_topology_summary.json`
- `face_branch_summary.json`
- `qualitative_case_notes.md`

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

Current local evaluation snapshot:

- paired clips actually present locally: `a1`, `a2`, `a3`, `b1`
- appearance-only pass count: `2`
- topology-supported pass count: `2`
- topology rescue count: `2`
- average `osnet_mean` body score on compared clips: `0.5942`
- average `osnet_quality_aware` body score on compared clips: `0.5939`
- average `osnet_x1_0_quality_aware` body score on compared clips: `0.5719`

Interpretation:

- topology rescue is still needed on the true physical `C1 -> C2` cross-view pair
- quality-aware pooling alone is not yet enough to push those body scores over `0.72`
- the stronger OSNet x1.0 benchmark does not currently beat the lighter extractor on average
