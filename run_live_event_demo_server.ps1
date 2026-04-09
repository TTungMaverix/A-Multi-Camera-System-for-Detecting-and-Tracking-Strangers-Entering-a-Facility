$env:PYTHONIOENCODING = 'utf-8'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptRoot '.venv_insightface_demo\Scripts\python.exe'
$runner = Join-Path $scriptRoot 'insightface_demo_assets\runtime\run_live_event_demo_server.py'
& $pythonExe $runner --project-root $scriptRoot --output-root 'outputs/live_runs/file_sanity'
