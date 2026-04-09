$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
powershell -ExecutionPolicy Bypass -File (Join-Path $scriptRoot 'run_offline_multicam_pipeline.ps1')
