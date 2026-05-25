@echo off
setlocal
cd /d "%~dp0"

if "%RUNTIME_PYTHON%"=="" set "RUNTIME_PYTHON=C:\Users\Admin\AppData\Local\Temp\ifd_venv\Scripts\python.exe"
if "%DATASET_ROOT%"=="" set "DATASET_ROOT=C:\Users\Admin\AppData\Local\Temp\ifd_dataset"
if "%KNOWN_DB_ROOT%"=="" set "KNOWN_DB_ROOT=C:\Users\Admin\AppData\Local\Temp\ifd_known"

if not exist "%RUNTIME_PYTHON%" (
  echo RUNTIME_PYTHON_MISSING=%RUNTIME_PYTHON%
  exit /b 1
)

"%RUNTIME_PYTHON%" ^
  "insightface_demo_assets\runtime\run_live_event_demo_server.py" ^
  --project-root "." ^
  --dataset-root "%DATASET_ROOT%" ^
  --demo-pair-id d2 ^
  --output-root "outputs\evaluations\d2_demo_artifacts\offline_runs\d2" ^
  --scene-calibration-config "insightface_demo_assets\runtime\config\manual_scene_calibration.d2.yaml" ^
  --stream-target-fps 15 ^
  --presentation-mode sequential ^
  --camera-sequence C1,C2,C3,C4 ^
  --camera-segment-sec auto ^
  --travel-gap-sec 8
