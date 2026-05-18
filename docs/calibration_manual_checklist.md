# Calibration Manual Checklist

Use this checklist every time a camera source changes or a new source camera is added.

## Rules

- Do not let Codex or any AI tool invent ROI coordinates.
- Do not copy a calibration from another source camera unless the geometry is truly identical.
- Do not trust a calibration until you verify the overlay on the real frame by eye.

## Manual Steps

1. Open the real video or stream frame for the target camera.
2. Draw the processing ROI polygon by hand.
3. Draw the entry line by hand.
4. Click the IN-side point by hand.
5. Save the YAML and JSON export from the calibration tool.
6. Run the pipeline or live server with that saved calibration file.
7. Inspect the overlay:
   - ROI covers only the intended region.
   - entry line sits on the true crossing boundary.
   - IN arrow points into the facility.
   - track footpoints and ENTRY_IN events align with the geometry.
8. If zones or subzones are needed, add them explicitly and review them on the frame.
9. Recalibrate if camera angle, crop, zoom, or frame geometry changes.

## Calibration Tool Controls

- Left click: add ROI points
- Right click or `Enter`: finish ROI
- Next 2 clicks: entry-line `p1`, `p2`
- Third click: IN-side point
- `u`: undo draft point
- `r`: reset current shape
- `s`: save
- `q` or `Esc`: quit

## Command Prompt Example

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" --camera C1 --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\a1.mp4" --output-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.custom.yaml" --frame-index 0
```

## Recalibrate B2 Camera 1

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" --camera C1 --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\b2.mp4" --output-config ".\outputs\evaluations\manual_calibration_smoke\manual_scene_calibration.C1_b2.yaml" --per-camera-json ".\outputs\evaluations\manual_calibration_smoke\C1_b2_config.json" --frame-index 0
```

## Recalibrate B2 Camera 2

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" --camera C2 --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 2\b2.mp4" --output-config ".\outputs\evaluations\manual_calibration_smoke\manual_scene_calibration.C2_b2.yaml" --per-camera-json ".\outputs\evaluations\manual_calibration_smoke\C2_b2_config.json" --frame-index 0
```

## Recalibrate C1 Camera 1

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" --camera C1 --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 1\c1.mp4" --output-config ".\outputs\evaluations\manual_calibration_smoke\manual_scene_calibration.C1_c1.yaml" --per-camera-json ".\outputs\evaluations\manual_calibration_smoke\C1_c1_config.json" --frame-index 0
```

## Recalibrate C1 Camera 2

```cmd
cd /d "D:\ĐỒ ÁN TỐT NGHIỆP"
".\.venv_insightface_demo\Scripts\python.exe" ".\tools\calibrate_camera.py" --camera C2 --source "D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Camera 2\c1.mp4" --output-config ".\outputs\evaluations\manual_calibration_smoke\manual_scene_calibration.C2_c1.yaml" --per-camera-json ".\outputs\evaluations\manual_calibration_smoke\C2_c1_config.json" --frame-index 0
```

## Verify Saved Calibration Through the UI

```cmd
cd /d "C:\Users\Admin\AppData\Local\Temp\doantn-p0-phase"
set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"
run_live_event_demo_server_b2.cmd
```

Then open:

- `http://127.0.0.1:8765`
- `http://127.0.0.1:8765/calibration.html`
