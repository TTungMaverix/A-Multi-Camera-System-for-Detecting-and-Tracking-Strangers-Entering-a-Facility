$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
powershell -ExecutionPolicy Bypass -File (Join-Path $scriptRoot 'run_single_source_sequential_video_phase.ps1')
