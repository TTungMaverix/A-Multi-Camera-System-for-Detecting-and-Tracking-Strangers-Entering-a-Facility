@echo off
setlocal

set "REPO=%~dp0"
cd /d "%REPO%"

if not defined PYTHON_EXE set "PYTHON_EXE=D:\ĐỒ ÁN TỐT NGHIỆP\.venv_insightface_demo\Scripts\python.exe"
if not defined DATASET_ROOT set "DATASET_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset"
if not defined KNOWN_DB_ROOT set "KNOWN_DB_ROOT=D:\ĐỒ ÁN TỐT NGHIỆP\New Dataset\Known ID"

if not exist ".\New Dataset\Camera 1\b2.mp4" (
  if exist "%DATASET_ROOT%\Camera 1\b2.mp4" (
    mklink /J "New Dataset" "%DATASET_ROOT%"
  )
)

"%PYTHON_EXE%" ".\insightface_demo_assets\runtime\run_live_event_demo_server.py" ^
  --project-root "." ^
  --dataset-root "%DATASET_ROOT%" ^
  --demo-pair-id b2 ^
  --output-root ".\outputs\evaluations\b2_known_face_runtime_validation\offline_runs\b2" ^
  --scene-calibration-config ".\insightface_demo_assets\runtime\config\manual_scene_calibration.new_dataset_demo.yaml" ^
  --stream-target-fps 15 ^
  --presentation-mode sequential ^
  --camera-sequence C1,C2,C3,C4 ^
  --camera-segment-sec 12 ^
  --travel-gap-sec 3

endlocal
