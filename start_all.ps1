$ErrorActionPreference = "Stop"

# ------------------------------------------------------------
# 0) Root & Settings
# ------------------------------------------------------------
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

# Haupt-venv (Backend, TTS)
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Error ".venv\Scripts\python.exe nicht gefunden. Bitte .venv anlegen und Dependencies installieren."
}

# Settings laden
$settingsPath = Join-Path $root "config\settings.json"
if (-not (Test-Path $settingsPath)) {
    Write-Error "config\settings.json fehlt."
}
$cfg = Get-Content $settingsPath -Raw | ConvertFrom-Json

# Logs-Ordner
if ($cfg.paths -and $cfg.paths.logs_dir) {
    $logsDir = Join-Path $root $cfg.paths.logs_dir
    if (-not (Test-Path $logsDir)) {
        New-Item -ItemType Directory -Path $logsDir | Out-Null
    }
}

$backendHost = if ($cfg.backend_host) { $cfg.backend_host } else { "127.0.0.1" }
$backendPort = if ($cfg.backend_port) { $cfg.backend_port } else { 8000 }

$uiHost      = if ($cfg.ui_host) { $cfg.ui_host } else { "127.0.0.1" }
$uiPort      = if ($cfg.ui_port) { $cfg.ui_port } else { $backendPort }
$openBrowser = $cfg.ui.open_browser

# ------------------------------------------------------------
# 1) Backend API (FastAPI)
# ------------------------------------------------------------
Write-Host "Starte Backend auf $backendHost`:$backendPort ..." -ForegroundColor Cyan

Start-Process -FilePath $venvPy `
    -WorkingDirectory $root `
    -ArgumentList @(
        "-m", "uvicorn", "backend.core:app",
        "--host", $backendHost,
        "--port", $backendPort.ToString()
    ) `
    -WindowStyle Minimized

# ------------------------------------------------------------
# 2) TTS-Server (optional)
# ------------------------------------------------------------
$media       = $cfg.media
$ttsEnabled  = $media.tts_enabled
$ttsPort     = if ($media.tts_port) { $media.tts_port } else { 8020 }
$ttsModelRel = $media.tts_model_path
$ttsModelAbs = if ($ttsModelRel) { Join-Path $root $ttsModelRel } else { $null }

if ($ttsEnabled -and $ttsModelAbs -and (Test-Path $ttsModelAbs)) {
    Write-Host "Starte TTS-Server auf Port $ttsPort ..." -ForegroundColor Cyan

    Start-Process -FilePath $venvPy `
        -WorkingDirectory $root `
        -ArgumentList @(
            "-m", "uvicorn", "backend.tts_server:app",
            "--host", "127.0.0.1",
            "--port", $ttsPort.ToString()
        ) `
        -WindowStyle Minimized
}
else {
    Write-Host "TTS deaktiviert oder Modell fehlt – TTS-Server wird nicht gestartet." -ForegroundColor Yellow
}

# ------------------------------------------------------------
# 3) ComfyUI (eigene venv + DirectML für Intel Arc)
# ------------------------------------------------------------
$comfyRoot   = Join-Path $root "ComfyUI"
$comfyVenvPy = Join-Path $comfyRoot "venv\Scripts\python.exe"

if ($media.comfy_enabled -and (Test-Path $comfyRoot) -and (Test-Path $comfyVenvPy)) {

    # Port aus comfy_host lesen (z.B. http://127.0.0.1:8002)
    $comfyHost = $media.comfy_host
    try {
        $uri = [System.Uri]$comfyHost
        if ($uri.Port -gt 0) {
            $comfyPort = $uri.Port
        }
        else {
            $comfyPort = 8188
        }
    }
    catch {
        $comfyPort = 8188
    }

    Write-Host "Starte ComfyUI auf Port $comfyPort (DirectML / Intel Arc)..." -ForegroundColor Cyan
    Write-Host "ComfyUI-venv: $comfyVenvPy" -ForegroundColor DarkGray
    Write-Host "torch-directml ist dort ja bereits installiert." -ForegroundColor DarkGray

    Start-Process -FilePath $comfyVenvPy `
        -WorkingDirectory $comfyRoot `
        -ArgumentList @(
            "main.py",
            "--listen", "127.0.0.1",
            "--port", $comfyPort.ToString(),
            "--directml",
            "--highvram"
        ) `
        -WindowStyle Minimized
}
else {
    Write-Host "ComfyUI ist in settings.json deaktiviert oder ComfyUI\venv fehlt – ComfyUI wird nicht gestartet." -ForegroundColor Yellow
}

# ------------------------------------------------------------
# 4) Browser auf UI öffnen
# ------------------------------------------------------------
if ($openBrowser) {
    $url = "http://$uiHost`:$uiPort/"
    Write-Host "Öffne Browser auf $url ..." -ForegroundColor Green
    Start-Process $url
}

Write-Host ""
Write-Host "Fertig. Backend, (optional) TTS und ComfyUI laufen jetzt." -ForegroundColor Green
Write-Host "Bei Problemen in die jeweiligen Fenster schauen." -ForegroundColor Green
