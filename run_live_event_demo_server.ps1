$env:PYTHONIOENCODING = 'utf-8'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptRoot '.venv_insightface_demo\Scripts\python.exe'
$runner = Join-Path $scriptRoot 'insightface_demo_assets\runtime\run_live_event_demo_server.py'
$sceneCalibration = 'insightface_demo_assets/runtime/config/manual_scene_calibration.wildtrack_4cam_phase.yaml'
$outputRoot = 'outputs/offline_runs/wildtrack_4cam_inference_roi_benchmark'
& $pythonExe $runner --project-root $scriptRoot --output-root $outputRoot --scene-calibration-config $sceneCalibration
