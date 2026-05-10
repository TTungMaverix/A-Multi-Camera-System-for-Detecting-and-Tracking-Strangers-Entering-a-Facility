# Project Constraints and Runtime Memory

This file records P0 constraints that must stay true across future phases.

## Scope

- Thesis topic: A Multi-Camera System for Detecting and Tracking Strangers Entering a Facility.
- Required pipeline: Detect -> Track -> Direction IN -> Face Known DB -> Unknown ID -> Cross-camera Association -> Event/timeline/UI evidence.
- Do not create logical replay events in API/UI when the pipeline failed to generate them.

## Active Dataset

- Dataset root: `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset`
- Camera 1: `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1`
- Camera 2: `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 2`
- Known DB root: `D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID`

## Logical Camera Mapping

- `C1`: physical Camera 1.
- `C2`: physical Camera 2.
- `C3`: delayed logical replay of Camera 1.
- `C4`: delayed logical replay of Camera 2.

`C3` and `C4` are demo replay streams, but direction analysis must run on their replayed tracks. `ENTRY_IN` must appear in pipeline event artifacts before the API/UI exposes it.

## Calibration Reuse

- Calibration is per source camera, not per clip.
- Reuse Camera 1 calibration for all clips captured from Camera 1 when camera position, angle, zoom, crop, resolution, and background geometry are unchanged.
- Reuse Camera 2 calibration for all clips captured from Camera 2 under the same conditions.
- Re-draw calibration only after a real geometry change.

## Direction Detector Rule

Direction detection uses a hierarchy:

1. Primary line crossing from OUT side to IN side.
2. ROI entry transition from outside ROI to inside ROI.
3. Clipped-track fallback: a track that first appears already inside the entry ROI/zone can emit `ENTRY_IN` only when it has enough detections, sufficient confidence, inside persistence, and inward/persistent motion evidence.

The clipped-track fallback reason code is `entry_inferred_from_inside_roi_track_start`. It is part of direction analysis, not API event fabrication.

## Known DB and Face Matching

- Active Known DB is under `New Dataset\Known ID`, not the legacy demo gallery.
- Build command:
  `python insightface_demo_assets/runtime/run_build_known_facility_db.py --known-root "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"`
- Face embedding cosine remains the primary known-face evidence when a usable face exists.
- Grayscale aligned face similarity is auxiliary evidence for lighting/domain-gap diagnostics.
- Never compare raw unaligned full images pixel-by-pixel.

## Re-ID Score Formula

Association decisions must expose:

`Score = a * FaceScore + b * BodyScore + c * TimeScore + d * TopologyScore`

Current new-dataset weights are configured in `association_policy.new_dataset_demo.yaml`.
Face is primary when usable; body is fallback/supporting evidence; time/topology are filters/support.

## Live Demo Server

- API serving must not decode video per frame request.
- Background workers decode/annotate frames and push latest JPEG/state into `FrameBufferHub`.
- `/api/camera-frame` returns latest buffered JPEG.
- `/api/camera-stream` and `/stream/camera/{camera_id}.mjpg` provide MJPEG streaming.
- Browser disconnects from MJPEG are expected and handled without traceback spam.
