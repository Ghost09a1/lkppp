# MyCandyLocal

Offline-first erotic roleplay stack for Windows. Runs a local FastAPI backend, static UI, optional llama.cpp GGUF models, Ollama fallback, SD.Next for NSFW image generation, and a stub TTS server prepared for mOrpheus GGUF.

## Features
- Local characters with builder UI and SQLite persistence.
- ERP-friendly persona compiler baked into system prompts (no censorship).
- Chat with streaming or single-shot replies via llama.cpp server (GGUF) or Ollama.
- Short-term history plus auto-summaries stored in SQLite (RAG-ready stubs).
- Image generation via SD.Next `/sdapi/v1/txt2img` (graceful fallback when offline).
- Local TTS HTTP service stub (returns silence if no model) ready for mOrpheus GGUF.
- Launcher for one-click start of services and UI.

## Layout
```
MyCandyLocal/
  app_launcher.py          # starts llama.cpp (optional), TTS server, backend
  config/settings.json     # ports, model paths, toggles
  backend/                 # FastAPI app + helpers
  ui/                      # static front-end (no Node needed)
  models/llm|tts|images|video/ # place your local models here
  bins/                    # place llama-server.exe and other binaries here
  outputs/                 # SQLite DB and generated files
  logs/                    # service logs
```

## Quickstart (Python 3.10+)
1) Create venv and install deps:
```powershell
cd MyCandyLocal
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r backend/requirements.txt
```

2) Place models/binaries:
- `bins/llama-server.exe` (llama.cpp server build for Windows/Arc/DirectML)  
- `models/llm/model.gguf` (uncensored ERP-capable GGUF)  
- Optional TTS: `models/tts/morpheus.gguf` (or leave missing; TTS auto-disables)  
- Optional SD.Next: run your local SD.Next on `http://127.0.0.1:7860`

3) Adjust `config/settings.json` if needed (ports, threads, ollama mode, enable/disable TTS/SD).

4) Start everything:
```powershell
python app_launcher.py
```
The launcher opens the UI (`/ui`). If llama.cpp binary/model are missing it will start in degraded mode (no LLM). TTS is skipped if the model is missing. Image generation calls SD.Next only when enabled.

## Running pieces manually
- Backend API only: `uvicorn backend.core:app --host 127.0.0.1 --port 8000`
- TTS server stub: `uvicorn backend.tts_server:app --port 8020`

## SQLite data
Stored at `outputs/chat.db` (characters, messages, summaries). Delete the file to reset history.

## Building app.exe (onefile)
1) Install PyInstaller: `pip install pyinstaller`
2) Build:
```powershell
pyinstaller --onefile --add-data "ui;ui" --add-data "config;config" app_launcher.py
```
3) Place `app.exe` next to `backend/`, `ui/`, `config/`, `models/`, `bins/`, `outputs/`, `logs/`.

## Notes & TODOs
- TTS currently returns silence when the GGUF model is missing; hook llama.cpp TTS or other engines at `backend/tts_server.py`.
- Video route is stubbed in `backend/media.py`; wire Wan2.2 worker later.
- RAG interface can be added by extending `memory.py` with a vector DB (Chroma/FAISS) without touching routes.
- Ensure no external network calls at runtime—only local services on loopback are used.
