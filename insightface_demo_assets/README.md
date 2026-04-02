# InsightFace Demo Assets

This folder now contains a working face-resolution stage for the graduation-project demo.

It plugs into the Wildtrack data-prep flow at:

- `D:\ĐỒ ÁN TỐT NGHIỆP\wildtrack_demo\output\events\identity_resolution_queue.csv`

## Current Status

- dedicated venv created: `D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo`
- runtime dependencies installed
- local patched InsightFace runtime is active inside the venv
- `buffalo_l` model pack downloaded to `C:\Users\Admin\.insightface\models\buffalo_l`
- face resolution demo script runs successfully on the prepared Wildtrack queue

## Main Outputs

- `runtime\known_face_embeddings_template.csv`
- `runtime\resolved_events_template.csv`
- `runtime\unknown_profiles_template.csv`
- `runtime\stream_identity_timeline.csv`
- `runtime\face_resolution_summary.json`
- `known_face_manifest_runtime.csv`

## Demo Behavior

1. Track a person in entry cameras `C5` / `C6` using the Wildtrack-prepared queue.
2. Run face detection and embedding on `best_head_crop`, then fallback to `best_body_crop`.
3. Compare against the known-face gallery.
4. If matched over threshold, assign `Known_ID`.
5. Otherwise assign `Unknown_Global_ID`.
6. Propagate the resolved identity across camera streams in the timeline export.

## Important Demo Note

For this dataset-only demo, cross-stream propagation is currently bridged by Wildtrack `global_gt_id`.
This lets you demonstrate the central identity-association flow now.
For the full project, replace that bridge with the real body re-id / topology / spatio-temporal association core.

## Run

Face-resolution step only:

```powershell
powershell -ExecutionPolicy Bypass -File .\insightface_demo_assets\runtime\run_face_resolution_demo.ps1
```

Full demo flow:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_multicam_identity_demo.ps1
```
