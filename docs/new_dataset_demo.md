# New Dataset Demo Profile

## Purpose

This document explains how the self-recorded New Dataset is connected to the existing thesis pipeline without rewriting the core algorithmic blocks.

## Physical Dataset Layout

Current physical folders:

- `New Dataset/Camera 1`
- `New Dataset/Camera 2`

Current pairing convention:

- `Camera 1/a1.mp4` pairs with `Camera 2/a1.mp4`
- future clips should keep the same shared-stem rule: `a2`, `b1`, ...

The runtime adapter discovers clip pairs by stem intersection across the two physical camera folders. It does not hardcode the runtime to `a1`.

## Logical 4-Camera Demo Expansion

The thesis scope stays at 4-camera demonstration level, but the active dataset currently has only 2 physical cameras.

The repo therefore uses explicit logical demo streams:

- `C1` -> physical Camera 1, offset `0s`
- `C2` -> physical Camera 2, offset `10s`
- `C3` -> replayed logical copy of physical Camera 1, offset `20s`
- `C4` -> replayed logical copy of physical Camera 2, offset `30s`

This is a documented demo adapter for scope alignment only.

## Camera Geometry Assumptions

- Camera 1:
  - top-down-ish diagonal view
  - face and full body are usually visible
  - ROI anchor mode defaults to `bottom_center`
- Camera 2:
  - more eye-level diagonal view
  - partial-body / upper-body detection is more common
  - ROI anchor mode defaults to `center_center`

The point-in-polygon math itself is unchanged. Only the anchor point used before the polygon test is configurable per camera.

## Topology

The New Dataset is treated as **non-overlap oriented**.

Association therefore emphasizes:

- reachable camera pairs
- min/max travel time windows
- face-first matching
- unknown profile reuse

Current default travel-time assumptions:

- `C1 -> C2`: `8s .. 16s`
- `C2 -> C3`: `8s .. 20s` logical demo continuation
- `C3 -> C4`: `8s .. 16s`

## Files

- dataset profile:
  - `insightface_demo_assets/runtime/config/dataset_profile.new_dataset_demo.yaml`
- scene calibration:
  - `insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml`
- topology:
  - `insightface_demo_assets/runtime/config/camera_transition_map.new_dataset_demo.yaml`
- policy:
  - `insightface_demo_assets/runtime/config/association_policy.new_dataset_demo.yaml`
- offline demo config:
  - `insightface_demo_assets/runtime/config/offline_pipeline_demo.new_dataset_logical_4cam_demo.yaml`

## Current Status

This phase prepares the framework so new self-recorded clips can enter the pipeline through configuration.

It does not claim:

- that the self-recorded dataset already contains 4 independent physical cameras
- that the current clip coverage is enough for a full benchmark phase
- that the Wildtrack benchmark/evaluation assets have been deleted from the repository
