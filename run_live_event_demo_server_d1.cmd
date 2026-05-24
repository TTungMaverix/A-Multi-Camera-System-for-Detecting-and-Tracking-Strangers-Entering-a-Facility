@echo off
setlocal
cd /d "%~dp0"

if "%DATASET_ROOT%"=="" set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
if "%KNOWN_DB_ROOT%"=="" set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"

"D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe" ^
  "insightface_demo_assets\runtime\run_live_event_demo_server.py" ^
  --project-root "." ^
  --dataset-root "%DATASET_ROOT%" ^
  --demo-pair-id d1 ^
  --output-root "outputs\evaluations\d1_demo_artifacts\offline_runs\d1" ^
  --scene-calibration-config "insightface_demo_assets\runtime\config\manual_scene_calibration.d1.yaml" ^
  --stream-target-fps 15 ^
  --presentation-mode sequential ^
  --camera-sequence C1,C2,C3,C4 ^
  --camera-segment-sec 12 ^
  --travel-gap-sec 8
