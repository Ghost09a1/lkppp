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

    # Befehl, der in der neuen PowerShell ausgeführt wird
    $psCommand = "Set-Location '$WorkingDirectory'; & '$python' $Arguments"

    Write-Host "Starting $Name in separate PowerShell window..."
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", $psCommand
}

# --- LLM server (llama.cpp) -------------------------------------------------

$llamaExe = Join-Path $root "bins\llama-server.exe"
if (Test-Path $llamaExe) {
    $llamaDir   = Split-Path $llamaExe -Parent
    $modelPath  = Join-Path $root "models\llm\model.gguf"

    if (-not (Test-Path $modelPath)) {
        Write-Warning "LLM model not found at '$modelPath' - LLM server will probably exit immediately."
    }

    # Hier Host/Port ggf. an deine config/settings.json anpassen!
    $llamaArgs = @(
        "--host", "127.0.0.1",
        "--port", "8080"
    )
    if (Test-Path $modelPath) {
        $llamaArgs = @("-m", $modelPath) + $llamaArgs
    }

    Write-Host "Starting LLM server (llama-server.exe)..." -ForegroundColor Green
    Start-Process -FilePath $llamaExe `
                  -WorkingDirectory $llamaDir `
                  -ArgumentList $llamaArgs
} else {
    Write-Host "LLM server binary bins\llama-server.exe not found, skipping." -ForegroundColor DarkYellow
}


# 1) MyCandyLocal launcher (starts backend, TTS, etc.)
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
Write-Host "Done. Launched all available servers (each in its own PowerShell window)."
