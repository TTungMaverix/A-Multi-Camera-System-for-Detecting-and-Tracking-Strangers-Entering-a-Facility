$env:PYTHONIOENCODING = 'utf-8'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptRoot '.venv_insightface_demo\Scripts\python.exe'
$runner = Join-Path $scriptRoot 'insightface_demo_assets\runtime\run_live_multicam_demo.py'
$config = Join-Path $scriptRoot 'insightface_demo_assets\runtime\config\live_pipeline_demo.file_sanity.yaml'
& $pythonExe $runner --config $config
