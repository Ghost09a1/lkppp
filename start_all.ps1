# stop on errors
$ErrorActionPreference = "Stop"

# repo root (folder where this script lives)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Repo root: $root"

# --------------------------------------------------------------------
# Root venv (MyCandyLocal)
# --------------------------------------------------------------------
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
    Write-Host "Using root venv Python: $python"
} else {
    $python = "python"
    Write-Host "Using system Python for root apps"
}

# --------------------------------------------------------------------
# LLM server (llama.cpp) based on settings.json
# --------------------------------------------------------------------
$llamaExe  = Join-Path $root "bins\llama-server.exe"
$modelPath = Join-Path $root "models\llm\MN-12B-Celeste-V1.9-Q4_K_M.gguf"

if (Test-Path $llamaExe) {
    if (-not (Test-Path $modelPath)) {
        Write-Warning "LLM model not found at '$modelPath' - LLM server may fail to start."
    }

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
# Helper: start apps in separate PowerShell windows
# --------------------------------------------------------------------
function Start-App {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [string]$PythonPath,
        [string]$Arguments
    )

    if (-not (Test-Path -LiteralPath $WorkingDirectory)) {
        Write-Warning "Directory '$WorkingDirectory' for $Name not found - skipping."
        return
    }

    $psCommand = "Set-Location '$WorkingDirectory'; & '$PythonPath' $Arguments"

    Write-Host "Starting $Name in separate PowerShell window..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $psCommand
}

# 1) MyCandyLocal launcher (Backend + TTS + UI etc. in root venv)
Start-App -Name "MyCandyLocal Launcher" -WorkingDirectory $root -PythonPath $python -Arguments "'app_launcher.py'"

# 2) ComfyUI mit eigenem venv
$comfyDir    = Join-Path $root "ComfyUI"
$comfyMain   = Join-Path $comfyDir "main.py"
$comfyPython = Join-Path $comfyDir "venv\Scripts\python.exe"  # ComfyUI venv heißt "venv"

if (Test-Path $comfyMain) {
    if (-not (Test-Path $comfyPython)) {
        Write-Warning "ComfyUI venv Python not found at '$comfyPython' - falling back to root Python."
        $comfyPython = $python
    } else {
        Write-Host "ComfyUI venv detected: $comfyPython"
    }

    Start-App -Name "ComfyUI" -WorkingDirectory $comfyDir -PythonPath $comfyPython -Arguments "'main.py'"
} else {
    Write-Host "ComfyUI not found, skipping."
}

# 3) RVC WebUI mit eigenem venv
$rvcDir    = Join-Path $root "rvc_webui"
$rvcMain   = Join-Path $rvcDir "infer-web.py"
$rvcPython = Join-Path $rvcDir "venv\Scripts\python.exe"  # rvc_webui venv heißt "venv"

if (Test-Path $rvcMain) {
    if (-not (Test-Path $rvcPython)) {
        Write-Warning "RVC venv Python not found at '$rvcPython' - falling back to root Python."
        $rvcPython = $python
    } else {
        Write-Host "RVC venv detected: $rvcPython"
    }

    Start-App -Name "RVC WebUI" -WorkingDirectory $rvcDir -PythonPath $rvcPython -Arguments "'infer-web.py'"
} else {
    Write-Host "RVC WebUI not found, skipping."
}

Write-Host ""
Write-Host "Done. Launched LLM server, MyCandyLocal, ComfyUI and RVC (if present)."
