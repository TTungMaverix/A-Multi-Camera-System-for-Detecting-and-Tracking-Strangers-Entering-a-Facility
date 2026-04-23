$env:PYTHONIOENCODING = 'utf-8'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptRoot '.venv_insightface_demo\Scripts\python.exe'
$runner = Join-Path $scriptRoot 'insightface_demo_assets\runtime\run_live_event_demo_server.py'
$sceneCalibration = 'insightface_demo_assets/runtime/config/manual_scene_calibration.new_dataset_demo.yaml'
$outputRoot = 'outputs/offline_runs/new_dataset_logical_4cam_demo'
& $pythonExe $runner --project-root $scriptRoot --output-root $outputRoot --scene-calibration-config $sceneCalibration
