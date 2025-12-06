# MyCandyLocal 🍬

Your private, uncensored AI companion.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React)                               │
│                        http://127.0.0.1:8000/ui                             │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  CharacterPanel │ ChatPanel │ MediaPanel │ CharacterEditor           │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ HTTP/SSE
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (FastAPI)                                 │
│                        http://127.0.0.1:8000                                │
│   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌─────────────┐  │
│   │   core.py      │ │   llm.py       │ │   media.py     │ │ memory.py   │  │
│   │  Main Router   │ │  LLM Client    │ │  TTS/Image     │ │ Chat Store  │  │
│   └────────┬───────┘ └───────┬────────┘ └───────┬────────┘ └─────────────┘  │
└────────────┼─────────────────┼──────────────────┼───────────────────────────┘
             │                 │                  │
             │                 ▼                  ▼
             │    ┌────────────────────┐   ┌──────────────────────────────────┐
             │    │   LLAMA.CPP        │   │         EXTERNAL SERVICES        │
             │    │   (LLM Server)     │   │                                  │
             │    │  :8080             │   │  ┌───────────────────────────┐   │
             │    │                    │   │  │ TTS Server (:8020)        │   │
             │    │  Model:            │   │  │ - pyttsx3 (fallback)      │   │
             │    │  L3-8B-Stheno      │   │  │ - Orpheus via LM Studio   │   │
             │    │  (Text Only!)      │   │  │   (:1234)                 │   │
             │    └────────────────────┘   │  └───────────────────────────┘   │
             │                             │  ┌───────────────────────────┐   │
             │                             │  │ ComfyUI (:8188)           │   │
             │                             │  │ - PonyDiffusion V6        │   │
             │                             │  │ - IPAdapter (consistency) │   │
             │                             │  └───────────────────────────┘   │
             │                             └──────────────────────────────────┘
             ▼
      ┌─────────────────┐
      │   SQLite DB     │
      │  characters.db  │
      │  + JSON memory  │
      └─────────────────┘
```

---

## 🧠 Models & Their Roles

| Component | Model | Location | Purpose |
|-----------|-------|----------|---------|
| **Chat LLM** | `L3-8B-Stheno-v3.2-Q5_K_M.gguf` | `models/llm/` | Text generation (NO SNAC tokens!) |
| **TTS (Orpheus)** | `Orpheus-3b-German-FT-Q8_0.gguf` | `models/tts/` | Text → SNAC → Audio (via LM Studio) |
| **TTS Vocoder** | `WavTokenizer-Large-75-F16.gguf` | `models/tts/` | SNAC token → waveform |
| **Image Gen** | PonyDiffusion V6 XL | ComfyUI `models/` | Text → Image (SDXL) |
| **STT** | Whisper | External | Speech → Text |

---

## 🔊 TTS Pipeline

### Architektur (Option A - Clean Text):

```
┌─────────────────┐          ┌─────────────────────────────────────┐
│   Stheno LLM    │          │           TTS Server (:8020)        │
│   (Text Only)   │          │                                     │
│                 │  CLEAN   │  ┌────────────────────────────────┐ │
│  "Das ist ein   │──TEXT───▶│  │ pyttsx3 (System TTS)           │ │
│   Test."        │          │  │ Text → WAV                     │ │
│                 │          │  └────────────────────────────────┘ │
│  *stöhnt leise* │          │                 ↓                   │
│  (Emotes als    │          │  ┌────────────────────────────────┐ │
│   Text, werden  │          │  │ RVC (Optional)                 │ │
│   für TTS       │          │  │ Voice Conversion → Nayuta      │ │
│   entfernt)     │          │  └────────────────────────────────┘ │
└─────────────────┘          └─────────────────────────────────────┘
```

### ⚠️ WICHTIG:

- **Stheno generiert KEINE SNAC-Tokens!** Es schreibt nur sauberen Text.
- Emotes in `*Sternchen*` werden für TTS entfernt, aber im Chat angezeigt.
- SNAC-Architektur wurde verworfen (LM Studio kann Tokens nicht zu Audio decodieren).

### TTS Server Dateien:

| Datei | Funktion |
|-------|----------|
| `backend/tts_server.py` | FastAPI TTS-Endpunkt, orchestriert alles |
| `backend/snac_tokenizer.py` | SNAC-Token-Decoder (Token → Audio) |
| `models/tts/gguf_orpheus.py` | Orpheus-Inference via LM Studio API |
| `models/tts/decoder.py` | (FEHLT?) SNAC → Waveform Konverter |

---

## 🎙️ Voice Training (RVC)

### Pipeline:

```
┌─────────────────┐     ┌─────────────────────────────────────┐
│  Character Edit │     │         Voice Training              │
│                 │     │                                     │
│  Upload MP3/WAV │────▶│  1. vc_train_tool.py               │
│  (Voice Sample) │     │     - Extracts audio features      │
│                 │     │     - Trains RVC model             │
│  Click "Train"  │     │                                     │
│                 │     │  2. Saves to:                       │
└─────────────────┘     │     outputs/models/{char_id}.pth   │
                        └─────────────────────────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────────┐
                        │         TTS + RVC Inference         │
                        │                                     │
                        │  pyttsx3 → WAV → rvc_infer.py      │
                        │                   ↓                 │
                        │              Character Voice        │
                        └─────────────────────────────────────┘
```

### Backend Dateien:

| Datei | Funktion |
|-------|----------|
| `backend/core.py` | `/characters/{id}/train_voice` Endpoint |
| `backend/core.py` | `_run_training()` - Training-Orchestrierung |
| `rvc_infer.py` | Voice Conversion Inference |
| `vc_train_tool.py` | Training Script (benötigt RVC WebUI) |

### Konfiguration (`config/settings.json`):

```json
{
  "media": {
    "rvc_cli_path": "path/to/rvc_cli.py",
    "rvc_webui_dir": "path/to/Retrieval-based-Voice-Conversion-WebUI",
    "vc_train_script": "vc_train_tool.py",
    "vc_script": "rvc_infer.py"
  }
}
```

### Training Status:

| Status | Bedeutung |
|--------|-----------|
| `queued` | Wartet auf Start |
| `running` | Training läuft |
| `done` | Fertig, Modell bereit |
| `failed` | Fehler (siehe `voice_error`) |

---

## 🖼️ Image Generation Pipeline

```
Chat Request → LLM → "[GENERATE_IMAGE] girl, blue eyes, ..."
                              │
                              ▼
                    ┌─────────────────┐
                    │  media.py       │
                    │  extract_prompt │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   ComfyUI API   │
                    │   :8188         │
                    │   - PonyV6 XL   │
                    │   - LoRAs       │
                    │   - IPAdapter   │
                    └────────┬────────┘
                             │
                             ▼
                       Base64 Image
                       → Frontend
```

---

## 📁 Projektstruktur

```
MyCandyLocal/
├── backend/
│   ├── core.py              # Main FastAPI app, routes
│   ├── llm.py               # LLM client (llama.cpp)
│   ├── media.py             # TTS/Image/Video orchestration
│   ├── memory.py            # Chat history, summarization
│   ├── tts_server.py        # TTS FastAPI (port 8020)
│   └── snac_tokenizer.py    # SNAC token decoder
├── frontend_v2/             # React app (source)
├── ui_v2/                   # Built frontend (served by backend)
├── models/
│   ├── llm/                 # Stheno, other chat models
│   └── tts/                 # Orpheus, WavTokenizer, gguf_orpheus.py
├── config/
│   └── settings.json        # All configuration
├── logs/                    # backend.log, backend_app.log, tts_app.log
├── outputs/                 # Generated images, avatars
├── start_all.ps1            # Starte alles
└── debug_restart.ps1        # Killt hängende Prozesse
```

---

## 🔧 Konfiguration (`config/settings.json`)

```json
{
  "llm": {
    "llama_cpp_url": "http://127.0.0.1:8080",
    "model_path": "models/llm/L3-8B-Stheno-v3.2-Q5_K_M.gguf"
  },
  "media": {
    "tts_enabled": true,
    "tts_port": 8020,
    "snac_model_id": "hubertsiuzdak/snac_24khz",
    "comfyui_url": "http://127.0.0.1:8188"
  }
}
```

---

## 🚀 Startup-Reihenfolge

1. **LLM Server** (llama.cpp) → Port 8080
2. **LM Studio** (Orpheus) → Port 1234 (optional, für Orpheus TTS)
3. **TTS Server** → Port 8020
4. **ComfyUI** → Port 8188 (optional, für Bilder)
5. **Backend** (FastAPI) → Port 8000
6. **Frontend** → http://127.0.0.1:8000/ui

---

## 🐛 Bekannte Probleme & Lösungen

| Problem | Ursache | Lösung |
|---------|---------|--------|
| "Wörter ohne Leerzeichen" | LLM (Stheno) versucht SNAC-Tokens zu generieren | SNAC-Anweisungen aus System-Prompt entfernen |
| Kein Audio | TTS Server nicht gestartet / LM Studio nicht mit Orpheus geladen | `start_all.ps1`, LM Studio mit Orpheus starten |
| Kein Bild | ComfyUI nicht gestartet / falscher Workflow | ComfyUI starten, Workflow prüfen |
| Tokens im Chat | Frontend-Filter fehlt/kaputt | `cleanDisplayText()` in `client.ts` prüfen |

---

## 📚 Wichtige Code-Stellen

| Funktion | Datei | Zeile(n) |
|----------|-------|----------|
| System Prompt | `backend/llm.py` | `_system_prompt()` ~50-110 |
| Chat Streaming | `backend/core.py` | `event_generator()` ~640-750 |
| Token Filter (Display) | `frontend_v2/src/api/client.ts` | `cleanDisplayText()` ~40-50 |
| TTS Entscheidung | `backend/core.py` | `if payload.enable_tts...` ~669 |
| SNAC Decoder | `backend/snac_tokenizer.py` | `decode_audio_from_ids()` |

---

## Credits

- Frontend: React + Vite + Tailwind
- Backend: FastAPI + Python
- LLM: Llama.cpp (Stheno 8B)
- TTS: Orpheus 3B (via LM Studio) + pyttsx3 fallback
- Image Gen: ComfyUI (Pony V6)
