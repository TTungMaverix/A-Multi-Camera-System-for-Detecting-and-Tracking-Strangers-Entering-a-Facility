$env:PYTHONIOENCODING = 'utf-8'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptRoot '.venv_insightface_demo\Scripts\python.exe'
$runner = Join-Path $scriptRoot 'insightface_demo_assets\runtime\run_offline_multicam_pipeline.py'
$config = Join-Path $scriptRoot 'insightface_demo_assets\runtime\config\offline_pipeline_demo.example.yaml'
& $pythonExe $runner --config $config
