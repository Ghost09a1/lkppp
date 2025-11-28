$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (!(Test-Path $venvPy)) {
    Write-Error "Konnte .venv\Scripts\python.exe nicht finden. Bitte zuerst .venv anlegen/aktivieren."
}

$settingsPath = Join-Path $root "config\settings.json"
if (!(Test-Path $settingsPath)) {
    Write-Error "config/settings.json fehlt."
}
$cfg = Get-Content $settingsPath -Raw | ConvertFrom-Json

# Llama.cpp Server (gguf) falls konfiguriert und Dateien vorhanden
if ($cfg.llm.mode -eq "gguf") {
    $llamaBin = Join-Path $root $cfg.llm.llama_cpp_server.binary_path
    $llamaModel = Join-Path $root $cfg.llm.llama_cpp_server.model_path
    if ((Test-Path $llamaBin) -and (Test-Path $llamaModel)) {
        $port = $cfg.llm.llama_cpp_server.port
        $threads = $cfg.llm.llama_cpp_server.n_threads
        $ctx = $cfg.llm.llama_cpp_server.n_ctx
        $batch = $cfg.llm.llama_cpp_server.batch
        $gpu = $cfg.llm.llama_cpp_server.gpu_layers
        $cmd = "cd `"$root`"; `"$llamaBin`" -m `"$llamaModel`" --port $port --ctx-size $ctx --threads $threads --batch-size $batch --host 127.0.0.1"
        if ($gpu -gt 0) { $cmd += " -ngl $gpu" }
        Start-Process powershell -ArgumentList @("-NoExit", "-Command", $cmd)
    } else {
        Write-Warning "llama.cpp binary oder Modell fehlt: $llamaBin / $llamaModel (LLM wird ausfallen)."
    }
} else {
    Write-Host "LLM mode = ollama (stelle sicher, dass der Ollama-Dienst läuft)." -ForegroundColor Yellow
}

# Image Generation Check
if ($cfg.media.image_mode -eq "sdnext") {
    Write-Host "Image generation mode is 'sdnext'. Make sure the SD.Next server is running on $($cfg.media.sdnext_host)." -ForegroundColor Yellow
} elseif ($cfg.media.image_mode -eq "local") {
    $sdxlModelPath = Join-Path $root $cfg.media.sdxl_model_path
    if (!(Test-Path $sdxlModelPath)) {
        Write-Warning "Image generation mode is 'local' but the model was not found at $sdxlModelPath. Image generation will fail."
    } else {
        Write-Host "Image generation mode is 'local'. The model will be loaded by the backend on first use." -ForegroundColor Green
    }
}


# Backend (API) -> use core (DB, chat, TTS routes)
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$root`"; `"$venvPy`" -m uvicorn backend.core:app --host 0.0.0.0 --port 8000"
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

Write-Host "Gestartet: Backend (8000), TTS (8020), ggf. llama.cpp ($($cfg.llm.llama_cpp_server.port))." -ForegroundColor Green
Write-Host "Falls Ports belegt sind, Prozesse beenden (node/python) und erneut ausführen."
