$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $venvPy)) {
    Write-Error "Konnte .venv\Scripts\python.exe nicht finden. Bitte zuerst .venv anlegen/aktivieren."
}

# Backend (API)
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$root`"; `"$venvPy`" backend/main.py"
)

# TTS-Server
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$root`"; `"$venvPy`" backend/tts_server.py"
)

# Frontend: Wir serven das gebaute UI über das Backend (Port 8000), kein Vite-Devserver nötig.
# Falls noch alte node-Prozesse laufen, beenden.
$nodeProcs = Get-Process node -ErrorAction SilentlyContinue
if ($nodeProcs) { $nodeProcs | Stop-Process -Force -ErrorAction SilentlyContinue }

# Browser öffnen (Backend liefert das UI)
Start-Process "http://localhost:8000/"

Write-Host "Gestartet: Backend (8000), TTS (8020), Frontend Dev (5173)." -ForegroundColor Green
Write-Host "Falls Ports belegt sind, Prozesse beenden (node/python) und erneut ausführen."
