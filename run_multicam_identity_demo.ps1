$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
powershell -ExecutionPolicy Bypass -File (Join-Path $scriptRoot 'run_wildtrack_4cam_roi_benchmark.ps1')
