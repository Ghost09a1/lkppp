# MyCandyLocal

Offline-first erotic roleplay stack for Windows. Runs a local FastAPI backend, modern vanilla JS UI, optional llama.cpp GGUF models, Ollama fallback, SD.Next/ComfyUI for NSFW image generation, and a local TTS server.

## Features
- **Modern Web UI**: Responsive dark-theme interface with character gallery, chat, and settings (vanilla HTML/CSS/JS - no build required)
- **Character Management**: Create, edit, and manage AI companions with personalities, backstories, and avatars
- **Smart Chat**: Streaming or single-shot replies via llama.cpp server (GGUF) or Ollama with conversation history
- **Image Generation**: Optional NSFW image generation via SD.Next or ComfyUI integration
- **Voice Synthesis**: Local TTS HTTP service for character voice playback
- **Memory System**: Short-term history plus auto-summaries stored in SQLite (RAG-ready stubs)
- **One-Click Launcher**: Start all services and open UI with a single command

## Layout
```
MyCandyLocal/
  app_launcher.py          # starts services and opens browser
  start_all.ps1            # (optional) starts llama.cpp server first
  config/settings.json     # ports, model paths, toggles
  backend/                 # FastAPI app + helpers
  ui/                      # modern vanilla JS front-end
    ├── index.html         # main UI structure
    ├── styles.css         # dark theme styling
    └── app.js             # full application logic
  models/llm|tts|images|video/ # place your local models here
  bins/                    # place llama-server.exe and other binaries here
  outputs/                 # SQLite DB and generated files
  logs/                    # service logs
  tests/                   # pytest test suite
```

## Quickstart (Python 3.10+)

### 1. Create venv and install deps:
```powershell
cd MyCandyLocal
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r backend/requirements.txt
```

### 2. Place models/binaries:
- **LLM (Required for chat)**:
  - Option A: `bins/llama-server.exe` + `models/llm/model.gguf` (uncensored ERP-capable GGUF)
  - Option B: Install and run Ollama, then edit `config/settings.json` to set `"mode": "ollama"`
- **TTS (Optional)**: `models/tts/morpheus.gguf` or leave missing (TTS auto-disables)
- **Images (Optional)**: Run SD.Next on `http://127.0.0.1:7860` or ComfyUI on port 8002

### 3. Configure settings (optional):
Edit `config/settings.json` to adjust:
- Ports and hosts
- LLM mode (gguf vs ollama)
- Feature toggles (TTS, image generation)
- Model paths

### 4. Start everything:

**Option A: With llama.cpp server**
```powershell
# Terminal 1: Start llama.cpp server
cd bins
.\\llama-server.exe -m ../models/llm/your-model.gguf -c 8192 -ngl 0 --port 8081

# Terminal 2: Start app
cd ..
python app_launcher.py
```

**Option B: With start_all.ps1 (if available)**
```powershell
.\\start_all.ps1
python app_launcher.py
```

**Option C: With Ollama**
```powershell
# Make sure Ollama is running
# Edit config/settings.json: "mode": "ollama"
python app_launcher.py
```

The launcher will:
- Start the FastAPI backend on port 8000
- Start TTS server (if enabled)
- Open your browser to `http://127.0.0.1:8000/ui/`

### 5. Use the UI:
1. **Create a Character**: Click "+ New Character", fill in details, upload avatar (optional)
2. **Start Chatting**: Go to "Chat" tab, select character, type messages
3. **Generate Images**: Click "🎨 Image" button in chat (requires SD.Next/ComfyUI running)
4. **Check Status**: Go to "Settings" tab to see which services are online

## Running pieces manually
- Backend API only: `uvicorn backend.core:app --host 127.0.0.1 --port 8000`
- TTS server stub: `uvicorn backend.tts_server:app --port 8020`
- LLM server: `bins/llama-server.exe -m models/llm/model.gguf --port 8081`

## Testing
Run backend tests:
```powershell
pytest tests/test_api.py -v
```

## SQLite data
Stored at `outputs/chat.db` (characters, messages, summaries). Delete the file to reset history.

## API Endpoints

### Characters
- `GET /characters` - List all characters
- `POST /characters` - Create new character
- `GET /characters/{id}` - Get character details
- `POST /characters/{id}/update` - Update character
- `POST /characters/{id}/avatar` - Upload avatar image

### Chat
- `POST /chat/{char_id}` - Send message and get reply (with optional TTS audio)
- `POST /chat_stream/{char_id}` - Streaming chat via Server-Sent Events

### Media
- `POST /stt` - Speech-to-text transcription
- `POST /tts` - Text-to-speech generation
- `POST /generate_image` - Generate image from prompt
- `POST /posts/{char_id}/image` - Generate and attach image to conversation

### System
- `GET /health` - Simple health check
- `GET /status` - Check status of backend, LLM, TTS, and image generation services

## Building app.exe (onefile)
1) Install PyInstaller: `pip install pyinstaller`
2) Build:
```powershell
pyinstaller --onefile --add-data "ui;ui" --add-data "config;config" app_launcher.py
```
3) Place `app.exe` next to `backend/`, `ui/`, `config/`, `models/`, `bins/`, `outputs/`, `logs/`

## Configuration Reference

### settings.json structure:
```json
{
  "backend_host": "127.0.0.1",
  "backend_port": 8000,
  "llm": {
    "mode": "gguf",  // or "ollama"
    "llama_cpp_server": {
      "binary_path": "bins/llama-server.exe",
      "model_path": "models/llm/model.gguf",
      "port": 8081,
      "n_ctx": 8192,
      "n_threads": 12,
      "gpu_layers": 0
    },
    "ollama": {
      "model": "model-name",
      "host": "http://127.0.0.1:11434"
    }
  },
  "media": {
    "tts_enabled": true,
    "tts_port": 8020,
    "sdnext_enabled": false,
    "sdnext_host": "http://127.0.0.1:7860",
    "comfy_enabled": false,
    "comfy_host": "http://127.0.0.1:8002"
  },
  "ui": {
    "open_browser": true,
    "dark_mode": true
  }
}
```

## Troubleshooting

### Backend won't start
- Check logs in `logs/backend_app.log`
- Verify all dependencies installed: `pip install -r backend/requirements.txt`
- Ensure outputs/, logs/ directories exist

### LLM not responding
- Verify llama.cpp server is running on port 8081 (or Ollama on 11434)
- Check Settings tab in UI for service status
- For llama.cpp: ensure model file exists at configured path
- For Ollama: ensure service is running and model is pulled

### Images not generating
- Verify SD.Next or ComfyUI is running on configured port
- Check `config/settings.json` has correct host/port
- Enable feature: `"sdnext_enabled": true` or `"comfy_enabled": true`

### TTS not working
- Check if TTS model exists at configured path
- Verify TTS server started (check logs/tts.log)
- Disable if not needed: `"tts_enabled": false`

## Notes & TODOs
- TTS currently supports mOrpheus GGUF models; hook llama.cpp TTS or other engines at `backend/tts_server.py`
- Video route is stubbed in `backend/media.py`; wire Wan2.2 worker later
- RAG interface can be added by extending `memory.py` with a vector DB (Chroma/FAISS)
- **Ensure no external network calls at runtime**—only local services on loopback are used

## License
Private/personal use. Check model licenses separately.
