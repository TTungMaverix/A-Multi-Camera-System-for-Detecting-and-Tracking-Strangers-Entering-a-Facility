# New Dataset Algorithmic Audit

This document records the audit that motivated the tracklet-body / strict-face / topology-signal phase, plus the current resolved status.

## Current Behavior

- Active offline entrypoint is `insightface_demo_assets/runtime/run_offline_multicam_pipeline.py` with config `insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml`.
- New Dataset is the active path.
- Local paired clips currently present are `a1`, `a2`, `a3`, and `b1`.
- `sequential.body_primary` is currently `0.72`, not `0.55`.
- Face evidence is selected from buffered head crops inside `run_face_resolution_demo.py` using explicit camera-role gating and strict size / pose rules.
- Body evidence is derived from a pooled tracklet representation rather than a single best crop.
- Topology / travel-time / zone / subzone are already used as hard filters before appearance similarity, and the decision trace now exposes topology-supported acceptance explicitly.

## Original Root Causes

These were the issues confirmed at the start of the earlier fix phase:

1. `sequential.body_primary = 0.55` was a threshold hack.
2. Directional filtering needed an independent validation artifact.
3. Body ReID was effectively frame-based.
4. The face branch needed a stricter camera-role policy.
5. Topology/time/zone/subzone needed to become a stronger decision signal for borderline sequential cases.

## Current Status After Fix

Current resolved points:

- the threshold hack is removed
- direction validation has a standalone run artifact
- body evidence is tracklet-based
- face capture is camera-role aware and quality-gated
- decision logs expose topology-supported acceptance explicitly

Current unresolved points:

- physical `C1 -> C2` body appearance is still not strong enough to pass by appearance alone on the current clips
- face evidence is still rare even on face-friendly cameras
- `a3` remains a hard failure case
- `a2` currently produces no entry events

## Files That Changed Across The Fix Phases

- `insightface_demo_assets/runtime/run_face_resolution_demo.py`
- `insightface_demo_assets/runtime/association_core/body_reid.py`
- `insightface_demo_assets/runtime/association_core/gallery_lifecycle.py`
- `insightface_demo_assets/runtime/association_core/appearance_evidence.py`
- `insightface_demo_assets/runtime/association_core/decision_policy.py`
- `insightface_demo_assets/runtime/association_core/quality_gate.py`
- `insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml`
- selected tests under `tests/`
- `README.md`
- selected docs under `docs/`

## Out Of Scope

- UI/dashboard work
- live RTSP-first deployment
- storage/backend redesign
- large model retraining
- FPS optimization before accuracy stabilization
- rewriting the core pipeline architecture
