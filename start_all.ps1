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

# 2) ComfyUI (optional)
$comfyDir = Join-Path $root "ComfyUI"
if (Test-Path (Join-Path $comfyDir "main.py")) {
    Start-App -Name "ComfyUI" -WorkingDirectory $comfyDir -Arguments "'main.py'"
} else {
    Write-Host "ComfyUI not found, skipping."
}

# 3) RVC WebUI (optional)
$rvcDir = Join-Path $root "rvc_webui"
if (Test-Path (Join-Path $rvcDir "infer-web.py")) {
    Start-App -Name "RVC WebUI" -WorkingDirectory $rvcDir -Arguments "'infer-web.py'"
} else {
    Write-Host "RVC WebUI not found, skipping."
}

Write-Host ""
Write-Host "Done. Launched LLM server and all available Python servers."
