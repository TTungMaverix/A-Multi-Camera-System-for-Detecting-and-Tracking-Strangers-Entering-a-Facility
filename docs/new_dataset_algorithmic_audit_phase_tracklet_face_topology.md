## Current Behavior

- Active offline entrypoint is `insightface_demo_assets/runtime/run_offline_multicam_pipeline.py`
  with `offline_pipeline/orchestrator.py` and config
  `insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml`.
- New Dataset is the active path. The current logical demo uses `a1` only; `a2` and `b1`
  are not present yet under `New Dataset/Camera 1` and `New Dataset/Camera 2`.
- Sequential association for the New Dataset currently uses
  `association_policy.new_dataset_demo.yaml`, where `relation_thresholds.sequential.body_primary`
  has been lowered to `0.55`.
- Face evidence is selected from buffered head crops inside `run_face_resolution_demo.py`
  using `evaluate_buffered_face_gate(...)`, but the current flow still evaluates all events
  rather than explicitly treating high-angle cameras as event/body anchors first.
- Body evidence is still derived from a single event crop (`best_body_crop`) rather than
  a pooled representation from multiple tracklet crops.
- Topology / travel-time / zone / subzone are already used as hard filters before
  appearance similarity, but the resulting decision trace still leans too heavily on raw
  appearance thresholds and does not yet present topology/time as the primary reasoning
  signal for borderline cases.

## Confirmed Root Causes

1. `sequential.body_primary = 0.55` is a threshold hack. It masks weak body evidence instead
   of improving the representation.
2. Directional filtering exists and has unit tests, but there is no independent run artifact
   that validates IN/OUT behavior on the active New Dataset path.
3. Body ReID is still effectively frame-based:
   one crop in, one embedding out, then compared across cameras.
4. The face branch correctly rejects many frames by pose/blur, but the active New Dataset
   still needs a stricter camera-role policy so that high-angle views do not implicitly act
   as valid face evidence sources.
5. Topology/time/zone/subzone are configured and enforced, but the decision logs and
   acceptance path do not yet make them a sufficiently strong decision signal relative to
   appearance evidence quality.

## Files Likely To Change

- `insightface_demo_assets/runtime/run_face_resolution_demo.py`
- `insightface_demo_assets/runtime/association_core/body_reid.py`
- `insightface_demo_assets/runtime/association_core/gallery_lifecycle.py`
- `insightface_demo_assets/runtime/association_core/appearance_evidence.py`
- `insightface_demo_assets/runtime/association_core/decision_policy.py`
- `insightface_demo_assets/runtime/association_core/quality_gate.py`
- `insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml`
- `tests/test_association_core.py`
- `tests/test_offline_pipeline.py`
- `tests/test_scene_calibration.py`
- `README.md`
- selected docs under `docs/`

## Out Of Scope For This Phase

- UI/dashboard work
- live RTSP-first deployment
- storage/backend redesign
- large model retraining
- FPS optimization before accuracy stabilization
- rewriting the core pipeline architecture
