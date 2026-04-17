# Quantitative Evaluation

## Purpose

This phase adds a small but reproducible quantitative evaluation package on top of the sequential replay benchmark.

The goal is not to claim a full public MTMCT benchmark result. The goal is to replace "looks correct" with:

- a fixed benchmark window
- a fixed GT/reference construction method
- exported metrics
- exported artifacts for threshold justification

## Benchmark Window

Current benchmark config:

- `insightface_demo_assets/runtime/config/offline_pipeline_demo.single_source_sequential_c6_inference_cache_benchmark.yaml`

Current evaluated clip:

- source video: `Wildtrack/cam6.mp4`
- replay mode: `C1 -> C2 -> C3 -> C4`
- actual frame range: `0 -> 720`
- actual duration: about `12.012s`
- stride: `6`

## Metrics

Two metrics are exported by default:

- `MOTA`
- `IDF1`

### MOTA

`MOTA` is measured on the first virtual replay camera only.

Why:

- sequential replay duplicates one physical source stream
- using one virtual camera avoids pretending the setup is a full public multi-view tracking benchmark
- it still gives a standard local detect+track quality number

GT for MOTA comes from Wildtrack annotations aligned to the actual source frames used by the cached/inference replay run.

### IDF1

`IDF1` is measured on direction-filtered replay events across `C1 -> C4`.

Why:

- the thesis demo logic is event-driven after direction filtering
- the sequential replay benchmark is specifically about keeping the same identity across virtual cameras

GT/reference for IDF1 is generated from the same replay config in explicit `gt_annotations` mode, then matched back to predicted events by:

- camera
- anchor frame tolerance
- anchor foot-point proximity from the first buffered observation

This is not advertised as a public MTMCT leaderboard result. It is a consistent internal benchmark for the thesis demo mode.

## Commands

Run the benchmark first:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_offline_multicam_pipeline.py" --config ".\insightface_demo_assets\runtime\config\offline_pipeline_demo.single_source_sequential_c6_inference_cache_benchmark.yaml"
```

Then run quantitative evaluation:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_quantitative_evaluation.py" --config ".\insightface_demo_assets\runtime\config\quantitative_evaluation.single_source_sequential_c6_cache_benchmark.yaml"
```

Then run threshold analysis:

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\insightface_demo_assets\runtime\run_threshold_analysis.py" --config ".\insightface_demo_assets\runtime\config\threshold_analysis.single_source_sequential_c6_cache_benchmark.yaml"
```

## Output Artifacts

Main evaluation files:

- `outputs/offline_runs/<run_name>/evaluation/quantitative_metrics_summary.json`
- `outputs/offline_runs/<run_name>/evaluation/pred_event_gt_matches.csv`
- `outputs/offline_runs/<run_name>/evaluation/pred_event_gt_unmatched_gt.csv`
- `outputs/offline_runs/<run_name>/evaluation/pred_event_gt_unmatched_pred.csv`
- `outputs/offline_runs/<run_name>/evaluation/identity_contingency.csv`

Threshold-analysis files:

- `outputs/offline_runs/<run_name>/evaluation/threshold_analysis/candidate_pair_dataset.csv`
- `outputs/offline_runs/<run_name>/evaluation/threshold_analysis/threshold_recommendation_summary.json`
- `outputs/offline_runs/<run_name>/evaluation/threshold_analysis/roc_curves.png`
- `outputs/offline_runs/<run_name>/evaluation/threshold_analysis/pr_curves.png`
- `outputs/offline_runs/<run_name>/evaluation/threshold_analysis/score_distributions.png`

## Threshold Recommendation Rule

The threshold script does not blindly pick the smallest PR threshold that reaches `F1=1`.

If positives and negatives are perfectly separated on a tiny pair set, it keeps a conservative threshold inside the safe gap instead.

That avoids fake "optimal" thresholds that look mathematically correct but are too lenient for a defense/demo pipeline.

## Timeline Artifact

The replay benchmark now also exports Unknown-ID timeline artifacts:

- `outputs/offline_runs/<run_name>/timelines/unknown_identity_timeline.csv`
- `outputs/offline_runs/<run_name>/timelines/unknown_identity_timeline.json`

These are consumed by the lightweight demo server and timeline UI.
