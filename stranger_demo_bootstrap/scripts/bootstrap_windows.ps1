param(
    [string]$ProjectRoot = ".",
    [string]$PythonExe = "",
    [string]$VenvName = ".venv_insightface_demo",
    [string]$ModelRoot = "",
    [string]$InsightFaceSource = "",
    [switch]$UseGpu,
    [switch]$SkipInstall,
    [switch]$SkipModelDownload,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PYTHONIOENCODING = "utf-8"

function Write-Step($msg) {
    Write-Host "[BOOTSTRAP] $msg" -ForegroundColor Cyan
}

function Ensure-Dir($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}

function Resolve-PythonExecutable([string]$RequestedPython) {
    $candidates = @()
    if ($RequestedPython) {
        $candidates += $RequestedPython
    }
    $candidates += @(
        (Join-Path $env:USERPROFILE ".platformio\python3\python.exe"),
        "py",
        "python"
    )

    foreach ($candidate in $candidates) {
        if (-not $candidate) {
            continue
        }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }
    throw "Khong tim thay Python that. Hay truyen -PythonExe hoac cai Python."
}

function Resolve-InsightFaceSourcePath([string]$RequestedPath, [string]$PackRoot) {
    $candidates = @()
    if ($RequestedPath) {
        $candidates += $RequestedPath
    }
    $candidates += @(
        (Join-Path $PackRoot "vendor\insightface_runtime_src\insightface"),
        (Join-Path $env:USERPROFILE "insightface-master\python-package\insightface")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "Khong tim thay source InsightFace local. Hay truyen -InsightFaceSource den thu muc chua package insightface."
}

function Copy-TemplateFile([string]$Source, [string]$Destination, [switch]$ForceCopy) {
    if ((Resolve-Path $Source).Path -eq (Resolve-Path $Destination -ErrorAction SilentlyContinue | ForEach-Object { $_.Path })) {
        return
    }
    if ((-not (Test-Path $Destination)) -or $ForceCopy) {
        Ensure-Dir (Split-Path -Parent $Destination)
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Install-PatchedInsightFace([string]$VenvPython, [string]$SourcePath) {
    Write-Step "Installing patched local InsightFace runtime"
    $sitePackages = & $VenvPython -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    $target = Join-Path $sitePackages "insightface"
    if (Test-Path $target) {
        Remove-Item -Recurse -Force $target
    }
    Copy-Item -Recurse -Force $SourcePath $target

    $rootInit = Join-Path $target "__init__.py"
@'
from . import model_zoo
from . import utils
from . import data
__version__ = "0.7.3-local"
'@ | Set-Content -Path $rootInit -Encoding UTF8

    $appInit = Join-Path $target "app\__init__.py"
@'
from .face_analysis import *
'@ | Set-Content -Path $appInit -Encoding UTF8
}

function Ensure-ModelPack([string]$VenvPython, [string]$ResolvedModelRoot, [switch]$SkipDownload) {
    $packPath = Join-Path $ResolvedModelRoot "models\buffalo_l"
    if (Test-Path $packPath) {
        Write-Step "Da tim thay model cache: $packPath"
        return
    }
    if ($SkipDownload) {
        Write-Warning "Chua co buffalo_l trong $ResolvedModelRoot va dang bo qua tai model."
        return
    }

    Write-Step "Thu khoi tao InsightFace de tai buffalo_l"
    $code = @"
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', root=r'$ResolvedModelRoot', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('MODEL_READY')
"@
    try {
        & $VenvPython -c $code
    }
    catch {
        Write-Warning "Khong tai duoc buffalo_l tu dong. Neu model chua co san, hay copy vao $packPath."
    }
}

$packRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path $ProjectRoot)) {
    New-Item -ItemType Directory -Path $ProjectRoot | Out-Null
}
$root = (Resolve-Path $ProjectRoot).Path
$resolvedPython = Resolve-PythonExecutable $PythonExe
if (-not $ModelRoot) {
    $ModelRoot = Join-Path $env:USERPROFILE ".insightface"
}

Write-Step "Project root: $root"
Write-Step "Using Python: $resolvedPython"
Write-Step "Model root: $ModelRoot"

$dirs = @(
    "config",
    "docs",
    "data",
    "data/streams",
    "data/known_db",
    "data/models",
    "data/outputs",
    "data/outputs/events",
    "data/outputs/snapshots",
    "data/outputs/clips",
    "data/outputs/logs",
    "data/outputs/debug",
    "data/cache",
    "data/db",
    "scripts",
    "src",
    "src/api",
    "src/config",
    "src/core",
    "src/ui",
    "tests",
    "vendor"
)

foreach ($d in $dirs) {
    Ensure-Dir (Join-Path $root $d)
}

$templatePairs = @(
    @{ Source = (Join-Path $packRoot "README.md"); Destination = (Join-Path $root "README.md") },
    @{ Source = (Join-Path $packRoot "requirements-demo.txt"); Destination = (Join-Path $root "requirements-demo.txt") },
    @{ Source = (Join-Path $packRoot "docs\CODEX_IMPLEMENTATION_BRIEF.md"); Destination = (Join-Path $root "docs\CODEX_IMPLEMENTATION_BRIEF.md") },
    @{ Source = (Join-Path $packRoot "docs\CODEX_PROMPT.txt"); Destination = (Join-Path $root "docs\CODEX_PROMPT.txt") },
    @{ Source = (Join-Path $packRoot "config\app.example.yaml"); Destination = (Join-Path $root "config\app.example.yaml") },
    @{ Source = (Join-Path $packRoot "config\cameras.example.yaml"); Destination = (Join-Path $root "config\cameras.example.yaml") },
    @{ Source = (Join-Path $packRoot "config\topology.example.yaml"); Destination = (Join-Path $root "config\topology.example.yaml") },
    @{ Source = (Join-Path $packRoot "scripts\bootstrap_windows.ps1"); Destination = (Join-Path $root "scripts\bootstrap_windows.ps1") },
    @{ Source = (Join-Path $packRoot "scripts\bootstrap_unix.sh"); Destination = (Join-Path $root "scripts\bootstrap_unix.sh") },
    @{ Source = (Join-Path $packRoot "scripts\build_known_db.py"); Destination = (Join-Path $root "scripts\build_known_db.py") },
    @{ Source = (Join-Path $packRoot "scripts\run_demo.py"); Destination = (Join-Path $root "scripts\run_demo.py") },
    @{ Source = (Join-Path $packRoot "src\__init__.py"); Destination = (Join-Path $root "src\__init__.py") },
    @{ Source = (Join-Path $packRoot "src\main.py"); Destination = (Join-Path $root "src\main.py") },
    @{ Source = (Join-Path $packRoot "src\config\__init__.py"); Destination = (Join-Path $root "src\config\__init__.py") },
    @{ Source = (Join-Path $packRoot "src\config\loader.py"); Destination = (Join-Path $root "src\config\loader.py") },
    @{ Source = (Join-Path $packRoot "src\core\__init__.py"); Destination = (Join-Path $root "src\core\__init__.py") },
    @{ Source = (Join-Path $packRoot "src\core\direction_filter.py"); Destination = (Join-Path $root "src\core\direction_filter.py") },
    @{ Source = (Join-Path $packRoot "src\core\topology.py"); Destination = (Join-Path $root "src\core\topology.py") },
    @{ Source = (Join-Path $packRoot "src\core\association.py"); Destination = (Join-Path $root "src\core\association.py") },
    @{ Source = (Join-Path $packRoot "src\core\event_logger.py"); Destination = (Join-Path $root "src\core\event_logger.py") },
    @{ Source = (Join-Path $packRoot "src\api\app.py"); Destination = (Join-Path $root "src\api\app.py") },
    @{ Source = (Join-Path $packRoot "src\ui\dashboard.py"); Destination = (Join-Path $root "src\ui\dashboard.py") },
    @{ Source = (Join-Path $packRoot "tests\test_topology.py"); Destination = (Join-Path $root "tests\test_topology.py") },
    @{ Source = (Join-Path $packRoot "tests\test_association.py"); Destination = (Join-Path $root "tests\test_association.py") }
)

foreach ($pair in $templatePairs) {
    Copy-TemplateFile -Source $pair.Source -Destination $pair.Destination -ForceCopy:$Force
}

Copy-TemplateFile -Source (Join-Path $root "config\app.example.yaml") -Destination (Join-Path $root "config\app.yaml") -ForceCopy:$Force
Copy-TemplateFile -Source (Join-Path $root "config\cameras.example.yaml") -Destination (Join-Path $root "config\cameras.yaml") -ForceCopy:$Force
Copy-TemplateFile -Source (Join-Path $root "config\topology.example.yaml") -Destination (Join-Path $root "config\topology.yaml") -ForceCopy:$Force

$venvPath = Join-Path $root $VenvName
if ((Test-Path $venvPath) -and $Force) {
    Write-Step "Removing existing virtual environment"
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Step "Creating virtual environment"
    & $resolvedPython -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

Write-Step "Ensuring pip tooling"
& $venvPython -m ensurepip --upgrade
& $venvPython -m pip --version | Out-Null

if (-not $SkipInstall) {
    Write-Step "Installing demo dependencies"
    & $venvPip install -r (Join-Path $root "requirements-demo.txt")

    if ($UseGpu) {
        Write-Step "Switching ONNXRuntime package to GPU variant"
        & $venvPip uninstall -y onnxruntime
        & $venvPip install onnxruntime-gpu==1.21.0
    }

    $resolvedInsightFaceSource = Resolve-InsightFaceSourcePath $InsightFaceSource $packRoot
    Install-PatchedInsightFace -VenvPython $venvPython -SourcePath $resolvedInsightFaceSource
    Ensure-ModelPack -VenvPython $venvPython -ResolvedModelRoot $ModelRoot -SkipDownload:$SkipModelDownload
}

$nextSteps = Join-Path $root "BOOTSTRAP_NEXT_STEPS.txt"
@"
1. Put your 4 videos or RTSP sources into config/cameras.yaml.
2. Put known identity images into data/known_db/<person_id>/.
3. Adjust config/topology.yaml to your map and travel-time windows.
4. Build known DB:
   .\$VenvName\Scripts\python.exe scripts\build_known_db.py
5. Validate the runtime:
   .\$VenvName\Scripts\python.exe scripts\run_demo.py
6. Optional API:
   .\$VenvName\Scripts\python.exe -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
7. Optional dashboard:
   .\$VenvName\Scripts\python.exe -m streamlit run src\ui\dashboard.py --server.port 8501
"@ | Set-Content -Path $nextSteps -Encoding UTF8

Write-Step "Bootstrap complete"
Write-Host "Virtual environment: $venvPath" -ForegroundColor Green
Write-Host "Config files are available under config/" -ForegroundColor Green
Write-Host "Next steps saved to BOOTSTRAP_NEXT_STEPS.txt" -ForegroundColor Green
