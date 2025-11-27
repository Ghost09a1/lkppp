import base64
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
import logging

import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, root_validator

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency
    sf = None

try:
    import pyttsx3
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None

try:
    import librosa
except Exception:  # pragma: no cover - optional dependency
    librosa = None
try:
    from .snac_tokenizer import decode_audio_from_text, extract_audio_token_ids, load_snac_model
except Exception:  # pragma: no cover - optional dependency
    decode_audio_from_text = None
    load_snac_model = None
    extract_audio_token_ids = None

# ---------------------------------------------------------
# Konfiguration laden
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "config" / "settings.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("mycandy.tts")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "tts_app.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

with SETTINGS_PATH.open("r", encoding="utf-8") as f:
    settings = json.load(f)

media_cfg = settings.get("media", {})
tts_enabled_cfg = bool(media_cfg.get("tts_enabled", False))
tts_model_rel = media_cfg.get("tts_model_path", "models/tts/tts-model.gguf")
tts_vocoder_rel = media_cfg.get("tts_vocoder_path", "")
tts_port = int(media_cfg.get("tts_port", 8020))
snac_model_id = media_cfg.get("snac_model_id", "hubertsiuzdak/snac_24khz")
snac_local_path = media_cfg.get("snac_local_path")
snac_device = media_cfg.get("snac_device", "cpu")
vc_script = media_cfg.get("vc_script", "")

MODEL_PATH = ROOT / tts_model_rel
VOCODER_PATH = ROOT / tts_vocoder_rel if tts_vocoder_rel else None
LLAMA_TTS_BIN = ROOT / "bins" / "llama-tts.exe"

TTS_AVAILABLE = (
    tts_enabled_cfg
    and MODEL_PATH.is_file()
    and LLAMA_TTS_BIN.is_file()
    and (VOCODER_PATH is None or VOCODER_PATH.is_file())
)

SNAC_DEPENDENCIES_OK = decode_audio_from_text is not None and load_snac_model is not None and sf is not None
SNAC_MODEL = None
TTS_ENGINE = None

EMOTE_REGEX = re.compile(r"\*[^*]+\*")
MARKERS_REGEX = re.compile(r"<\|audio_start\|>|<\|audio_end\|>")
CUSTOM_TOKEN_REGEX = re.compile(r"<custom_token_\d+>")

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------

app = FastAPI(title="MyCandyLocal TTS Server")


class TTSRequest(BaseModel):
    text: str | None = None
    message: str | None = None
    voice_style: str | None = None  # optional; spaeter nutzbar (soft, breathy, etc.)
    pitch_shift: float | None = 0.0
    speed: float | None = 1.0
    voice_ref_path: str | None = ""
    voice_model_path: str | None = ""

    @root_validator(pre=True)
    def merge_text_fields(cls, values):
        # allow either "text" or "message" from clients
        if not values.get("text") and values.get("message"):
            values["text"] = values["message"]
        return values


class TTSResponse(BaseModel):
    ok: bool
    audio_base64: str | None = None
    sample_rate: int | None = None
    error: str | None = None


@app.get("/health")
async def health():
    return {
        "ok": True,
        "tts_available": TTS_AVAILABLE,
        "model_path": str(MODEL_PATH),
        "vocoder_path": str(VOCODER_PATH) if VOCODER_PATH else "",
        "snac_available": SNAC_DEPENDENCIES_OK,
        "snac_model_loaded": SNAC_MODEL is not None,
        "snac_model_id": snac_model_id,
    }


def _ensure_snac_model():
    global SNAC_MODEL
    if SNAC_MODEL is None and SNAC_DEPENDENCIES_OK:
        SNAC_MODEL = load_snac_model(
            model_id=snac_model_id,
            local_path=snac_local_path,
            device=snac_device,
        )
    return SNAC_MODEL


def _wav_bytes_from_numpy(wav_np):
    if sf is None:
        raise RuntimeError("soundfile not available; install via `pip install soundfile`")
    buf = io.BytesIO()
    sf.write(buf, wav_np, 24000, format="WAV")
    buf.seek(0)
    return buf.read()


def _synth_pyttsx3(text: str) -> bytes:
    """
    Synthesize via pyttsx3 per request (fresh engine to avoid hangs).
    """
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 not available; install via `pip install pyttsx3 pypiwin32`")
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        out_path = tmp.name
    try:
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        data = Path(out_path).read_bytes()
    finally:
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            engine.stop()
        except Exception:
            pass
    return data


def _clean_for_plain_tts(text: str) -> str:
    cleaned = MARKERS_REGEX.sub("", text)
    cleaned = CUSTOM_TOKEN_REGEX.sub("", cleaned)
    cleaned = EMOTE_REGEX.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned or text


def _apply_pitch_speed(wav_bytes: bytes, pitch_shift: float, speed: float) -> bytes:
    if sf is None:
        return wav_bytes
    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return wav_bytes
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if librosa is not None:
        if pitch_shift and abs(pitch_shift) > 0:
            try:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            except Exception:
                pass
        if speed and speed != 1.0:
            try:
                audio = librosa.effects.time_stretch(audio, rate=speed)
            except Exception:
                pass
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _maybe_convert_voice(wav_bytes: bytes, voice_ref_path: str, voice_model_path: str) -> bytes:
    """
    Voice conversion hook. If a VC script is configured (media.vc_script), and both the
    reference sample and voice model path exist, call the script to convert the audio.

    Expected VC script signature (example, RVC-style):
      python <vc_script> --model <voice_model_path> --input <in.wav> --output <out.wav> [--ref <voice_ref_path>]
    """
    if not vc_script or not voice_model_path:
        return wav_bytes
    script_path = Path(vc_script)
    if not script_path.exists():
        return wav_bytes
    if not voice_ref_path or not Path(voice_ref_path).exists():
        return wav_bytes
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "vc_in.wav"
            out_path = Path(tmpdir) / "vc_out.wav"
            in_path.write_bytes(wav_bytes)
            cmd = [
                "python",
                str(script_path),
                "--model",
                str(voice_model_path),
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--ref",
                str(voice_ref_path),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if out_path.exists():
                return out_path.read_bytes()
    except Exception:
        return wav_bytes
    return wav_bytes


@app.post("/tts", response_model=TTSResponse)
async def tts(req: TTSRequest):
    """
    Nimmt Text entgegen, decodiert SNAC-Tokens wenn vorhanden,
    faellt sonst auf llama-tts zurueck.
    """
    text = (req.text or "").strip() if req.text else ""
    if not text:
        return TTSResponse(ok=False, error="Empty text")
    logger.info("tts recv len=%s has_tokens=%s", len(text), bool(extract_audio_token_ids and extract_audio_token_ids(text)))

    token_ids = extract_audio_token_ids(text) if extract_audio_token_ids else []

    pitch = req.pitch_shift or 0.0
    speed = req.speed or 1.0
    voice_ref = (req.voice_ref_path or "").strip()
    voice_model_path = (req.voice_model_path or "").strip()

    # --- Preferred path: SNAC decode when tokens exist ---
    if token_ids and SNAC_DEPENDENCIES_OK:
        try:
            model = _ensure_snac_model()
            wav_np = decode_audio_from_text(text_with_tokens=text, snac_model=model, device=snac_device)
            wav_bytes = _wav_bytes_from_numpy(wav_np)
            wav_bytes = _apply_pitch_speed(wav_bytes, pitch, speed)
            wav_bytes = _maybe_convert_voice(wav_bytes, voice_ref, voice_model_path)
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            logger.info("tts snac ok len=%s", len(b64))
            return TTSResponse(ok=True, audio_base64=b64, sample_rate=24000)
        except Exception as exc:
            snac_error = f"SNAC decode failed: {exc}"
    elif token_ids and not SNAC_DEPENDENCIES_OK:
        snac_error = "SNAC dependencies missing; install snac torch soundfile and restart."
    else:
        snac_error = None

    # --- Fallback 1: pyttsx3 text TTS ---
    if not token_ids and pyttsx3 is not None:
        try:
            clean_text = _clean_for_plain_tts(text)
            wav_bytes = _synth_pyttsx3(clean_text)
            wav_bytes = _apply_pitch_speed(wav_bytes, pitch, speed)
            wav_bytes = _maybe_convert_voice(wav_bytes, voice_ref, voice_model_path)
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            logger.info("tts pyttsx3 ok len=%s", len(b64))
            return TTSResponse(ok=True, audio_base64=b64, sample_rate=24000)
        except Exception:
            pass

    # --- Fallback 2: llama-tts binary ---
    if not TTS_AVAILABLE:
        return TTSResponse(
            ok=False,
            error=snac_error or "TTS not available (model or binary missing, or disabled in settings.json).",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.wav"

        clean_text = _clean_for_plain_tts(text)
        cmd = [
            str(LLAMA_TTS_BIN),
            "-m",
            str(MODEL_PATH),
            "-p",
            clean_text,
            "-o",
            str(out_path),
            "-ngl",
            "0",
        ]
        if VOCODER_PATH:
            cmd.extend(["-mv", str(VOCODER_PATH)])

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as e:
            return TTSResponse(ok=False, error=f"Failed to start llama-tts: {e}")

        if result.returncode != 0:
            return TTSResponse(
                ok=False,
                error=f"llama-tts exited with code {result.returncode}: {result.stderr}",
            )

        if not out_path.is_file():
            return TTSResponse(ok=False, error="Output WAV not found")

        data = out_path.read_bytes()
        data = _apply_pitch_speed(data, pitch, speed)
        data = _maybe_convert_voice(data, voice_ref, voice_model_path)
        b64 = base64.b64encode(data).decode("ascii")
        logger.info("tts llama-tts ok len=%s", len(b64))

    return TTSResponse(ok=True, audio_base64=b64, sample_rate=24000)


# ---------------------------------------------------------
# Direkter Start (z.B. python backend/tts_server.py)
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    import uvicorn

    # Ensure project root is importable when invoking as script
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=tts_port,
        reload=False,
    )
