$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
powershell -ExecutionPolicy Bypass -File (Join-Path $scriptRoot 'run_new_dataset_logical_demo.ps1')
