# Calibration Manual Checklist

Use this checklist every time a camera source changes or a new source camera is added.

## Rules

- Do not let Codex/AI invent ROI coordinates.
- Do not copy a calibration from another source camera unless geometry is truly the same.
- Do not trust a calibration until you verify the overlay on the real frame by eye.

## Manual Steps

1. Open the real video or stream frame for the target camera.
2. Draw the processing ROI polygon by hand.
3. Draw the entry line by hand.
4. Click the IN-side point by hand.
5. Save the YAML/JSON export from the calibration tool.
6. Run the pipeline or live server with that saved calibration file.
7. Inspect the overlay:
   - ROI covers only the intended region.
   - entry line sits on the true crossing boundary.
   - IN arrow points into the facility.
   - track footpoints and ENTRY_IN events align with the geometry.
8. If zones/subzones are needed, add them explicitly and review them on the frame.
9. Recalibrate if camera angle, crop, zoom, or frame geometry changes.

## Command Prompt Example

```cmd
cd /d "<repo-root>"
".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" --camera C1 --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\a1.mp4" --output-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.custom.yaml" --frame-index 0
```
