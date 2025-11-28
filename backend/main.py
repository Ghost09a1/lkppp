import base64
import json
import os
import time
import uuid
import wave
import subprocess
from pathlib import Path

import requests
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path when executed directly
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm import LLMClient
from backend.media import MediaRouter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SETTINGS_PATH = os.path.join(BASE_DIR, "config", "settings.json")

# Settings laden (falls vorhanden)
SETTINGS = {}
if os.path.exists(SETTINGS_PATH):
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        SETTINGS = json.load(f)

UI_DIR = os.path.join(BASE_DIR, SETTINGS.get("paths", {}).get("ui_dir", "ui"))

with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

app = FastAPI(title="MyCandyLocal API")

# CORS für das React-Frontend (Vite-Devserver)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statische Auslieferung generierter Audios
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=OUTPUTS_DIR), name="audio")
# Optional: serve model images if present
if os.path.exists(MODELS_DIR):
    app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")
from fastapi.responses import RedirectResponse

# Serve built frontend (unter /ui), Root -> Redirect
if UI_DIR and os.path.exists(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    assets_dir = Path(UI_DIR) / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir, html=False), name="ui-assets")


@app.get("/")
async def root_redirect():
    if UI_DIR and os.path.exists(UI_DIR):
        return RedirectResponse(url="/ui/")
    return {"status": "ok"}

media_router = MediaRouter(SETTINGS)
llm_client = LLMClient(SETTINGS)


def _list_models_with_meta():
    """Scannt models/ nach .pth und versucht zugehörige Bild/Meta zu finden."""
    models = []
    if not os.path.exists(MODELS_DIR):
        return models

    for f in os.listdir(MODELS_DIR):
        if not f.endswith(".pth"):
            continue
        stem = f.replace(".pth", "")
        # Suche nach optionalem Bild im gleichen Ordner
        img_path = ""
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = os.path.join(MODELS_DIR, f"{stem}{ext}")
            if os.path.exists(candidate):
                img_path = f"/models/{os.path.basename(candidate)}"
                break
        # optionale Metadata aus stem.json
        meta = {}
        meta_path = os.path.join(MODELS_DIR, f"{stem}.json")
        if os.path.exists(meta_path):
            try:
                meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        models.append(
            {
                "id": f,
                "name": stem,
                "image": img_path,
                "description": meta.get("description", "RVC Voice Model"),
                "personality": meta.get("personality", ""),
                "language": meta.get("language", ""),
            }
        )
    return models


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return cleaned or f"char_{uuid.uuid4().hex[:6]}"



def _maybe_call_ollama(prompt: str, model: str | None = None) -> str | None:
    """Versucht lokal Ollama anzusprechen; gibt None bei Fehler zurück."""
    ollama_model = model or os.environ.get("OLLAMA_MODEL", "")
    if not ollama_model:
        return None
    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response")
    except Exception as e:  # noqa: BLE001
        print(f"Ollama call failed: {e}")
        return None


def _write_silent_wav(path: str, duration_sec: float = 1.0, sample_rate: int = 16000):
    """Erzeugt eine kurze stille WAV, falls keine echte Audio-Pipeline angeschlossen ist."""
    n_frames = int(duration_sec * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


async def _transcribe_audio(audio_path: Path) -> str | None:
    """Versucht optional STT über faster-whisper oder whisper. Fällt bei Fehler auf None zurück."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        WhisperModel = None  # type: ignore

    if WhisperModel is not None:
        try:
            model_size = os.environ.get("WHISPER_MODEL", "small")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(audio_path), beam_size=1)
            text = "".join(seg.text for seg in segments)
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"Whisper STT failed: {exc}")
    return None


@app.get("/api/status")
async def get_status():
    return {"status": "online", "gpu_mode": os.environ.get("RVC_USE_DML", "0")}


@app.get("/api/characters")
async def get_characters():
    """Scannt den models/-Ordner nach .pth Dateien und liefert eine Liste mit Meta zurück."""
    return _list_models_with_meta()


@app.post("/api/generate_image")
async def generate_image(prompt: str = Form(...), negative: str = Form("")):
    img_resp = await media_router.generate_image(prompt, negative, 20, 512, 768)
    if not img_resp.get("ok"):
        raise HTTPException(status_code=500, detail=img_resp.get("error", "Image worker unavailable"))

    images = img_resp.get("images_base64") or img_resp.get("images") or []
    if not images:
        raise HTTPException(status_code=500, detail="Keine Bilder zurückgegeben.")
    return {"image_base64": images[0]}


@app.post("/api/characters/create")
async def create_character(
    name: str = Form(...),
    model_file: UploadFile | None = File(None),
    image_file: UploadFile | None = File(None),
):
    if model_file is None or not model_file.filename:
        raise HTTPException(status_code=400, detail="Model (.pth) wird benötigt.")

    safe_name = _safe_name(name)
    model_ext = Path(model_file.filename).suffix.lower()
    if model_ext != ".pth":
        raise HTTPException(status_code=400, detail="Bitte ein .pth Model hochladen.")

    model_path = Path(MODELS_DIR) / f"{safe_name}{model_ext}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(await model_file.read())

    if image_file and image_file.filename:
        img_ext = Path(image_file.filename).suffix.lower()
        if img_ext in {".png", ".jpg", ".jpeg", ".webp"}:
            img_path = Path(MODELS_DIR) / f"{safe_name}{img_ext}"
            img_path.write_bytes(await image_file.read())

    return {"id": model_path.name}


@app.post("/api/chat")
async def chat_endpoint(
    text: str = Form(""),
    character_id: str = Form(...),
    audio: UploadFile | None = File(None),
):
    """
    Pipeline: optional Audio speichern -> STT (wenn vorhanden) -> LLM -> TTS -> RVC (CLI) -> audio_url.
    """
    print(f"Chat Anfrage: '{text}' an Character '{character_id}'")

    # Speichere Upload, falls vorhanden
    upload_path = None
    if audio is not None:
        uploads_dir = Path(OUTPUTS_DIR) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        upload_path = uploads_dir / f"{int(time.time())}_{audio.filename}"
        with upload_path.open("wb") as f:
            f.write(await audio.read())

    # STT (optional)
    if not text and upload_path:
        text = await _transcribe_audio(upload_path)
        if not text:
            text = "(Audio empfangen)"

    # LLM-Antwort (Ollama/Llama.cpp aus settings)
    history = [{"role": "user", "content": text}]
    llm_text = await llm_client.generate_chat({"name": character_id}, history, stream=False)
    if not llm_text:
        llm_text = f"Ich habe '{text}' gehört! Ich bin {character_id}."

    # TTS -> WAV (über MediaRouter -> tts_server)
    tts_resp = await media_router.tts(
        text=llm_text,
        character={
            "name": character_id,
            "voice_model_path": str(Path(MODELS_DIR) / character_id),
        },
    )
    audio_filename = f"chat_{uuid.uuid4().hex}.wav"
    audio_path = Path(OUTPUTS_DIR) / audio_filename

    if tts_resp.get("ok") and tts_resp.get("audio_base64"):
        audio_bytes = base64.b64decode(tts_resp["audio_base64"])
        audio_path.write_bytes(audio_bytes)
    elif upload_path:
        audio_path.write_bytes(upload_path.read_bytes())
    else:
        _write_silent_wav(str(audio_path))

    # RVC-Konvertierung via rvc_infer wrapper (optional)
    model_path = Path(MODELS_DIR) / character_id
    rvc_cli = SETTINGS["media"].get("vc_script") or ""
    if model_path.exists() and rvc_cli:
        out_path = Path(OUTPUTS_DIR) / f"chat_rvc_{uuid.uuid4().hex}.wav"
        try:
            cmd = [
                "python",
                str(Path(BASE_DIR) / "rvc_infer.py"),
                "--model",
                str(model_path),
                "--input",
                str(audio_path),
                "--output",
                str(out_path),
                "--rvc_cli",
                str(Path(BASE_DIR) / SETTINGS["media"].get("rvc_cli_path", "rvc_cli.py")),
            ]
            subprocess.run(cmd, check=False)
            if out_path.exists():
                audio_path = out_path
                audio_filename = out_path.name
        except Exception as exc:  # noqa: BLE001
            print(f"RVC conversion failed: {exc}")

    return {
        "text": llm_text,
        "audio_url": f"/audio/{audio_filename}",
        "emotion": "neutral",
    }


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
