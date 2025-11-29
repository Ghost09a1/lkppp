# stop on errors
$ErrorActionPreference = "Stop"

# repo root (folder where this script lives)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Repo root: $root"

# prefer Python from venv
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
    Write-Host "Using venv Python: $python"
} else {
    $python = "python"
    Write-Host "Using system Python"
}

# --------------------------------------------------------------------
# LLM server (llama.cpp) nach settings.json
# --------------------------------------------------------------------
$llamaExe  = Join-Path $root "bins\llama-server.exe"
$modelPath = Join-Path $root "models\llm\MN-12B-Celeste-V1.9-Q4_K_M.gguf"

if (Test-Path $llamaExe) {
    if (-not (Test-Path $modelPath)) {
        Write-Warning "LLM model not found at '$modelPath' - LLM server may fail to start."
    }

    # Werte aus settings.json:
    # host: 127.0.0.1
    # port: 8081
    # n_ctx: 8192
    # n_threads: 12
    # gpu_layers: 0
    # batch: 128

    $llamaArgs = @(
        "--model", $modelPath,
        "--host", "127.0.0.1",
        "--port", "8081",
        "--ctx-size", "8192",
        "--threads", "12",
        "--n-gpu-layers", "0",
        "--batch-size", "128"
    )

    Write-Host "Starting LLM server (llama-server.exe) on port 8081..."
    Start-Process -FilePath $llamaExe -WorkingDirectory $root -ArgumentList $llamaArgs
} else {
    Write-Host "LLM server binary not found at '$llamaExe', skipping LLM server."
}

# --------------------------------------------------------------------
# Helper zum Start von Python-Apps in eigenen Fenstern (-NoExit)
# --------------------------------------------------------------------
function Start-App {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [string]$Arguments
    )

    if (-not (Test-Path -LiteralPath $WorkingDirectory)) {
        Write-Warning "Directory '$WorkingDirectory' for $Name not found - skipping."
        return
    }

    $psCommand = "Set-Location '$WorkingDirectory'; & '$python' $Arguments"

    Write-Host "Starting $Name in separate PowerShell window..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $psCommand
}

# 1) MyCandyLocal launcher (Backend + TTS + UI etc.)
Start-App -Name "MyCandyLocal Launcher" -WorkingDirectory $root -Arguments "'app_launcher.py'"

# 2) ComfyUI XPU Portable (Intel GPU)
$comfyDir      = Join-Path $root "ComfyUI"
$comfyLauncher = Join-Path $comfyDir "RUN_Launcher.bat"

if (Test-Path $comfyLauncher) {
    Write-Host "Starting ComfyUI XPU (portable)..."
    # .bat direkt starten, der Launcher kümmert sich um Python & Umgebung
    Start-Process -FilePath $comfyLauncher -WorkingDirectory $comfyDir
} else {
    Write-Host "ComfyUI launcher not found at '$comfyLauncher', skipping."
}

# 3) RVC WebUI (mit eigenem venv)
$rvcDir        = Join-Path $root "rvc_webui"
$rvcScript     = Join-Path $rvcDir "infer-web.py"
$rvcVenvPython = Join-Path $rvcDir "venv\Scripts\python.exe"

if (Test-Path $rvcScript) {
    if (Test-Path $rvcVenvPython) {
        $rvcPython = $rvcVenvPython
        Write-Host "Using RVC venv Python: $rvcPython"
    } else {
        $rvcPython = $python
        Write-Warning "RVC venv Python not found, falling back to: $rvcPython"
    }

    $rvcCommand = "Set-Location '$rvcDir'; & '$rvcPython' 'infer-web.py'"
    Write-Host "Starting RVC WebUI in separate PowerShell window..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $rvcCommand
} else {
    Write-Host "RVC WebUI not found, skipping."
}

Write-Host ""
Write-Host "Done. Launched LLM server and all available Python servers."
