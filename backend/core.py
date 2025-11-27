import json
import asyncio
from pathlib import Path
from typing import Any, Dict
import zipfile
import threading
import time
import subprocess
import shutil
import logging

from fastapi import FastAPI, HTTPException, File, UploadFile, Body, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import db
from .llm import LLMClient
from .memory import MemoryManager
from .media import MediaRouter

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("mycandy.core")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "backend_app.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_tts_text(text: str) -> str:
    """
    Prepare text for TTS:
    - strip audio markers
    - keep custom_token_* so the TTS backend can synthesize non-verbal sounds
    - skip emote text unless it contains custom_token_* (in which case keep those tokens)
    """
    import re

    without_markers = re.sub(r"<\|audio_start\|>|<\|audio_end\|>", "", text, flags=re.IGNORECASE)

    parts: list[str] = []
    last = 0
    # Preserve custom tokens from emotes; drop non-token emote text
    for match in re.finditer(r"\*(.*?)\*", without_markers):
        # text before emote
        parts.append(without_markers[last : match.start()])
        emote_body = match.group(1)
        tokens = re.findall(r"<custom_token_\d+>", emote_body, flags=re.IGNORECASE)
        if tokens:
            parts.append(" ".join(tokens))
        last = match.end()
    # tail after last emote
    parts.append(without_markers[last:])

    cleaned = "".join(parts)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip().strip('"').strip("'")


def _dedupe_reply(text: str) -> str:
    """
    Remove common duplication patterns the model sometimes returns:
    - drop content after the first ``` fence (often a full repeat)
    - collapse consecutive identical paragraphs/lines
    - limit repeated sentences and overall length for TTS sanity
    """
    import re

    if not text:
        return ""

    # Drop repeated block after ``` if present
    if "```" in text:
        head = text.split("```", 1)[0].strip()
        if head:
            text = head

    # Collapse consecutive identical lines/paragraphs
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    deduped_lines: list[str] = []
    for p in parts:
        if deduped_lines and p == deduped_lines[-1]:
            continue
        deduped_lines.append(p)
    text = "\n".join(deduped_lines)

    # Limit repeated sentences: allow max 2 occurrences of the same sentence
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen_counts: dict[str, int] = {}
    kept: list[str] = []
    for s in sentences:
        key = s.strip()
        if not key:
            continue
        seen_counts[key] = seen_counts.get(key, 0) + 1
        if seen_counts[key] <= 2:
            kept.append(key)
    text = " ".join(kept)

    # Hard cap length to avoid runaway repeats
    max_chars = 1600
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."

    return text.strip()


def _convert_to_wav(src: Path) -> Path | None:
    """
    Convert an audio file to wav using ffmpeg. Returns destination path or None on failure.
    """
    dst = src.with_suffix(".wav")
    try:
        # Downsample to 16k mono for more stable STT results
        cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0 and dst.exists():
            return dst
    except FileNotFoundError:
        return None
    return None


def _normalize_audio(src: Path) -> Path | None:
    """
    Loudness-normalize audio to improve STT on very quiet recordings.
    """
    dst = src.with_name(src.stem + "_norm.wav")
    try:
        # loudnorm keeps levels consistent without hard clipping
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            "loudnorm=I=-20:TP=-1.5:LRA=11",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dst),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0 and dst.exists():
            return dst
    except FileNotFoundError:
        return None
    return None


def _convert_dir_to_wav(raw_dir: Path) -> None:
    """
    Convert all supported compressed audio files in a directory to wav.
    Raises HTTPException if conversion fails for any mp3-like file.
    """
    for ext in (".mp3", ".m4a", ".ogg", ".flac"):
        for src in raw_dir.glob(f"**/*{ext}"):
            dst = _convert_to_wav(src)
            if dst is None:
                raise HTTPException(status_code=500, detail="ffmpeg missing or audio conversion failed.")
            try:
                src.unlink()
            except Exception:
                pass


class CharacterPayload(BaseModel):
    name: str
    description: str = ""
    visual_style: str = ""
    appearance_notes: str = ""
    personality: str = ""
    backstory: str = ""
    relationship_type: str = ""
    dos: str = ""
    donts: str = ""
    voice_style: str = ""
    voice_pitch_shift: float = 0.0
    voice_speed: float = 1.0
    voice_ref_path: str = ""
    voice_youtube_url: str = ""
    voice_model_path: str = ""
    voice_training_status: str = ""
    voice_error: str = ""
    language: str = "en"


class ChatPayload(BaseModel):
    message: str


class TtsPayload(BaseModel):
    message: str
    character_id: int | None = None


class ImagePayload(BaseModel):
    prompt: str
    negative: str = ""
    steps: int = 20
    width: int = 512
    height: int = 768


class VideoPayload(BaseModel):
    prompt: str


def create_app() -> FastAPI:
    config = load_config()
    db_path = str((ROOT / config["paths"]["db_path"]).resolve())
    db.init_db(db_path)
    conn = db.connect(db_path)
    avatars_dir = ROOT / "outputs" / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)
    media_cfg = config.get("media", {})
    vc_train_script = media_cfg.get("vc_train_script", "vc_train_tool.py")
    rvc_cli_path = media_cfg.get("rvc_cli_path", "")
    rvc_webui_dir = media_cfg.get("rvc_webui_dir", "")

    llm_client = LLMClient(config)
    memory = MemoryManager(conn, llm_client, config["chat"]["max_history"], config["chat"]["summary_every"])
    media = MediaRouter(config)

    app = FastAPI(title="MyCandyLocal")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ui_dir = ROOT / config["paths"]["ui_dir"]
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    app.mount("/avatars", StaticFiles(directory=str(avatars_dir), html=False), name="avatars")

    def _first_wav_in_raw(char_id: int) -> str:
        raw_dir = ROOT / "data" / "voices" / f"char_{char_id}" / "raw"
        if not raw_dir.exists():
            return ""
        candidates = sorted(raw_dir.glob("*.wav"))
        return str(candidates[0]) if candidates else ""

    def _save_avatar_file(char_id: int, upload: UploadFile, prev_path: str | None = None) -> str:
        # remove old avatar if present
        if prev_path:
            old = avatars_dir / prev_path
            if old.exists():
                try:
                    old.unlink()
                except Exception:
                    pass

        # enforce png output; unique filename to avoid cache
        filename = f"char_{char_id}_{int(time.time())}.png"
        out_path = avatars_dir / filename
        data = upload.file.read()
        try:
            from PIL import Image  # type: ignore
            import io

            img = Image.open(io.BytesIO(data)).convert("RGBA")
            img = img.resize((100, 100))
            img.save(out_path, format="PNG")
        except Exception:
            # fallback: just write raw bytes
            out_path.write_bytes(data)
        return filename

    def _set_training_status(char_id: int, status: str, error: str = "", model_path: str = ""):
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE characters
            SET voice_training_status = ?, voice_error = ?, voice_model_path = COALESCE(NULLIF(?, ''), voice_model_path)
            WHERE id = ?
            """,
            (status, error, model_path, char_id),
        )
        conn.commit()

    training_logs_dir = ROOT / "outputs" / "training"
    training_logs_dir.mkdir(parents=True, exist_ok=True)

    def _run_training(char_id: int):
        raw_dir = ROOT / "data" / "voices" / f"char_{char_id}" / "raw"
        model_out = ROOT / "models" / "vc" / f"char_{char_id}.pth"
        model_out.parent.mkdir(parents=True, exist_ok=True)
        log_path = training_logs_dir / f"char_{char_id}.log"
        if not raw_dir.exists() or not any(raw_dir.glob("*.wav")):
            _set_training_status(char_id, "failed", "No WAV files found in dataset.")
            return
        if not Path(vc_train_script).exists():
            _set_training_status(char_id, "failed", f"Training script not found: {vc_train_script}")
            return

        _set_training_status(char_id, "running", "")
        repo_path = rvc_webui_dir or rvc_cli_path
        if not repo_path or not Path(repo_path).exists():
            _set_training_status(char_id, "failed", "RVC repo path missing (set media.rvc_webui_dir).")
            return

        cmd = [
            "python",
            str(vc_train_script),
            "--data_dir",
            str(raw_dir),
            "--output",
            str(model_out),
            "--rvc_cli",
            str(repo_path),
        ]

        with log_path.open("w", encoding="utf-8") as lf:
            lf.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')} cmd: {' '.join(cmd)}\n")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            with log_path.open("a", encoding="utf-8") as lf:
                if proc.stdout:
                    for line in proc.stdout:
                        lf.write(line)
                        lf.flush()
            proc.wait()
            if proc.returncode == 0 and model_out.exists():
                _set_training_status(char_id, "done", "", str(model_out))
            else:
                _set_training_status(char_id, "failed", f"Exit {proc.returncode}")
        except Exception as exc:
            _set_training_status(char_id, "failed", str(exc))

    @app.get("/")
    async def root():
        return RedirectResponse(url="/ui/")

    @app.get("/characters")
    async def list_characters():
        return {"characters": db.list_characters(conn)}
    # Aliases for /api/* compatibility (legacy)
    @app.get("/api/characters")
    async def list_characters_api():
        return await list_characters()

    @app.post("/characters")
    async def create_character(payload: CharacterPayload):
        char_id = db.add_character(conn, payload.dict())
        return {"id": char_id}
    @app.post("/api/characters")
    async def create_character_api(payload: CharacterPayload):
        return await create_character(payload)

    @app.get("/characters/{char_id}")
    async def get_character(char_id: int):
        character = db.get_character(conn, char_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")
        return character
    @app.get("/api/characters/{char_id}")
    async def get_character_api(char_id: int):
        return await get_character(char_id)

    @app.post("/characters/{char_id}/update")
    async def update_character(char_id: int, payload: CharacterPayload):
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE characters
            SET name=:name, description=:description, visual_style=:visual_style, appearance_notes=:appearance_notes,
                personality=:personality, backstory=:backstory, relationship_type=:relationship_type,
                dos=:dos, donts=:donts, voice_style=:voice_style, voice_pitch_shift=:voice_pitch_shift,
                voice_speed=:voice_speed, voice_ref_path=:voice_ref_path, voice_youtube_url=:voice_youtube_url,
                voice_model_path=:voice_model_path, voice_training_status=:voice_training_status, voice_error=:voice_error,
                language=:language
            WHERE id=:id
            """,
            {**payload.dict(), "id": char_id},
        )
        conn.commit()
        return {"ok": True}
    @app.post("/api/characters/{char_id}/update")
    async def update_character_api(char_id: int, payload: CharacterPayload):
        return await update_character(char_id, payload)

    @app.post("/stt")
    async def stt(
        file: UploadFile = File(...),
        language: str = Form(""),
    ):
        if not file:
            raise HTTPException(status_code=400, detail="No audio provided.")
        uploads_dir = ROOT / "outputs" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        upload_path = uploads_dir / f"{int(time.time())}_{file.filename}"
        with upload_path.open("wb") as f:
            f.write(await file.read())

        stt_input = upload_path
        if upload_path.suffix.lower() != ".wav":
            wav_converted = _convert_to_wav(upload_path)
            if wav_converted is not None:
                stt_input = wav_converted
            else:
                logger.warning("stt convert failed for %s", upload_path)
        normalized = _normalize_audio(stt_input)
        if normalized is not None:
            stt_input = normalized

        lang_pref = (language or "").strip() or media.config.get("media", {}).get("whisper_language", "")
        transcript = await media.transcribe_audio(stt_input, language=lang_pref or None)
        if not transcript:
            raise HTTPException(status_code=500, detail="Transcription failed or empty.")
        return {"text": transcript, "language": lang_pref or ""}
    @app.post("/api/stt")
    async def stt_api(
        file: UploadFile = File(...),
        language: str = Form(""),
    ):
        return await stt(file, language)

    @app.post("/chat/{char_id}")
    async def chat(
        char_id: int,
        request: Request,
        payload: ChatPayload | None = Body(None),
        message: str = Form(""),
        audio: UploadFile | None = File(None),
    ):
        character = db.get_character(conn, char_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")

        # Accept message from JSON body or form field
        user_text = ((payload.message if payload else "") or message or "").strip()
        transcript: str | None = None

        # Fallback: if still empty, try raw JSON body
        if not user_text:
            try:
                raw = await request.body()
                if raw:
                    import json as _json

                    data = _json.loads(raw)
                    if isinstance(data, dict):
                        user_text = str(data.get("message", "") or data.get("text", "") or "").strip()
            except Exception:
                pass

        logger.info("chat recv char=%s len=%s audio=%s", char_id, len(user_text), bool(audio))

        if not user_text and audio is None:
            raise HTTPException(status_code=400, detail="Empty message.")

        # Handle optional audio -> save and transcribe (best-effort)
        if audio is not None:
            uploads_dir = ROOT / "outputs" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            upload_path = uploads_dir / f"{int(time.time())}_{audio.filename}"
            with upload_path.open("wb") as f:
                f.write(await audio.read())

            # Convert to wav for more reliable STT
            stt_input = upload_path
            if upload_path.suffix.lower() != ".wav":
                wav_converted = _convert_to_wav(upload_path)
                if wav_converted is not None:
                    stt_input = wav_converted
                else:
                    logger.warning("chat stt convert failed for %s", upload_path)
            # Normalize volume to avoid near-silent captures
            normalized = _normalize_audio(stt_input)
            if normalized is not None:
                stt_input = normalized

            lang_pref = (character.get("language") or "").strip()
            if not lang_pref:
                lang_pref = media.config.get("media", {}).get("whisper_language", "")

            if not user_text:
                transcript = await media.transcribe_audio(stt_input, language=lang_pref or None)
                user_text = transcript or "(audio received)"
            else:
                # audio present and text provided: keep a copy for transparency
                transcript = await media.transcribe_audio(stt_input, language=lang_pref or None)
            logger.info("chat stt transcript_len=%s", len(transcript or ""))
            if not transcript or not transcript.strip():
                user_text = "(audio not recognized)"

        memory.add_message(char_id, "user", user_text)
        history = memory.get_context(char_id)
        history.append({"role": "user", "content": user_text})
        fallback_msg = "LLM unavailable. Ensure llama.cpp server or Ollama is running locally."
        reply = None
        # retry loop to avoid showing fallback when the model is still warming up
        for attempt in range(3):
            reply = await llm_client.generate_chat(character, history, stream=False)
            if reply and reply != fallback_msg:
                break
            await asyncio.sleep(0.8)

        if reply is None or reply == fallback_msg:
            print("[chat] LLM returned None or fallback")
            reply = fallback_msg
        else:
            reply = _dedupe_reply(reply)
            preview = (reply[:120] + "...") if len(reply) > 120 else reply
            print(f"[chat] reply ok (len={len(reply)}) for char {char_id}: {preview}")
            logger.info("chat reply char=%s len=%s", char_id, len(reply))
        # Avoid polluting history with fallback error text
        if reply != fallback_msg:
            memory.add_message(char_id, "assistant", reply)
        await memory.maybe_summarize(char_id)
        # TTS uses cleaned text to avoid reading asterisks/markup
        tts_text = _clean_tts_text(reply)
        logger.info("chat tts_text_len=%s", len(tts_text))

        audio_b64: str | None = None
        try:
            tts_result = await media.tts(tts_text, character)
            if tts_result.get("ok") and tts_result.get("audio_base64"):
                audio_b64 = tts_result.get("audio_base64")
                logger.info("chat tts inline ok len=%s", len(audio_b64))
            else:
                logger.warning("chat tts inline failed: %s", tts_result.get("error"))
        except Exception as exc:
            logger.warning("chat tts inline exception: %s", exc)

        return {
            "reply": reply,
            "reply_tts": tts_text,
            "audio_base64": audio_b64,
            "user_text": user_text,
            "transcription": transcript or "",
        }
    @app.post("/api/chat/{char_id}")
    async def chat_api(char_id: int, payload: ChatPayload | None = Body(None), message: str = Form(""), audio: UploadFile | None = File(None)):
        return await chat(char_id, payload, message, audio)

    @app.post("/chat_stream/{char_id}")
    async def chat_stream(char_id: int, payload: ChatPayload):
        character = db.get_character(conn, char_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")
        memory.add_message(char_id, "user", payload.message)
        history = memory.get_context(char_id)
        history.append({"role": "user", "content": payload.message})
        stream = await llm_client.generate_chat(character, history, stream=True)
        if stream is None:
            async def error_stream():
                yield "data: LLM unavailable. Ensure backend model is running.\n\n"

            return EventSourceResponse(error_stream())

        async def event_generator():
            buffer = []
            async for token in stream:
                buffer.append(token)
                yield f"data: {token}\n\n"
            reply_text = "".join(buffer) if buffer else ""
            memory.add_message(char_id, "assistant", reply_text)
            await memory.maybe_summarize(char_id)
            yield "data: [STREAM_END]\n\n"

        return EventSourceResponse(event_generator())

    @app.post("/generate_image")
    async def generate_image(payload: ImagePayload):
        return await media.generate_image(payload.prompt, payload.negative, payload.steps, payload.width, payload.height)

    @app.post("/generate_video")
    async def generate_video(payload: VideoPayload):
        return await media.generate_video(payload.prompt)

    @app.post("/tts")
    async def tts(payload: TtsPayload):
        character = db.get_character(conn, payload.character_id) if payload.character_id else None
        return await media.tts(payload.message, character)
    @app.post("/api/tts")
    async def tts_api(payload: TtsPayload):
        return await tts(payload)

    @app.post("/characters/{char_id}/voice_sample")
    async def upload_voice_sample(char_id: int, file: bytes = None):
        if file is None:
            raise HTTPException(status_code=400, detail="No file provided")
        voices_dir = ROOT / "outputs" / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        out_path = voices_dir / f"char_{char_id}.wav"
        with open(out_path, "wb") as f:
            f.write(file)
        cur = conn.cursor()
        cur.execute("UPDATE characters SET voice_ref_path = ? WHERE id = ?", (str(out_path), char_id))
        conn.commit()
        return {"ok": True, "path": str(out_path)}
    @app.post("/api/characters/{char_id}/voice_sample")
    async def upload_voice_sample_api(char_id: int, file: bytes = None):
        return await upload_voice_sample(char_id, file)

    @app.post("/characters/{char_id}/avatar")
    async def upload_avatar(char_id: int, file: UploadFile = File(...)):
        character = db.get_character(conn, char_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")
        if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
            raise HTTPException(status_code=400, detail="Only PNG/JPEG images are allowed")
        prev = character.get("avatar_path") or None
        filename = _save_avatar_file(char_id, file, prev_path=prev)
        cur = conn.cursor()
        cur.execute("UPDATE characters SET avatar_path = ? WHERE id = ?", (filename, char_id))
        conn.commit()
        return {"ok": True, "avatar_path": filename, "url": f"/avatars/{filename}"}

    @app.post("/characters/{char_id}/voice_sample_url")
    async def upload_voice_sample_url(char_id: int, payload: Dict[str, str]):
        url = payload.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")
        voices_dir = ROOT / "outputs" / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        out_path = voices_dir / f"char_{char_id}.wav"
        # download audio via yt-dlp as wav
        try:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "-x",
                    "--audio-format",
                    "wav",
                    "-o",
                    str(out_path),
                    url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="yt-dlp not installed on server.")

        if result.returncode != 0 or not out_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Download failed: {result.stderr}",
            )
        cur = conn.cursor()
        cur.execute(
            "UPDATE characters SET voice_ref_path = ?, voice_youtube_url = ? WHERE id = ?",
            (str(out_path), url, char_id),
        )
        conn.commit()
        return {"ok": True, "path": str(out_path)}
    @app.post("/api/characters/{char_id}/voice_sample_url")
    async def upload_voice_sample_url_api(char_id: int, payload: Dict[str, str]):
        return await upload_voice_sample_url(char_id, payload)

    @app.post("/characters/{char_id}/voice_dataset")
    async def upload_voice_dataset(char_id: int, file: UploadFile = File(...)):
        raw_dir = ROOT / "data" / "voices" / f"char_{char_id}" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        filename = file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        if suffix == ".zip":
            tmp_zip = raw_dir / "_tmp_upload.zip"
            with tmp_zip.open("wb") as f:
                f.write(await file.read())
            try:
                with zipfile.ZipFile(tmp_zip, "r") as zf:
                    zf.extractall(raw_dir)
            finally:
                tmp_zip.unlink(missing_ok=True)
            _convert_dir_to_wav(raw_dir)
            saved_files = [str(p) for p in raw_dir.glob("**/*.wav")]
        else:
            temp_target = raw_dir / filename
            with temp_target.open("wb") as f:
                f.write(await file.read())
            if suffix in {".mp3", ".m4a", ".ogg", ".flac"}:
                wav_target = _convert_to_wav(temp_target)
                if wav_target is None:
                    raise HTTPException(status_code=500, detail="ffmpeg missing or audio conversion failed.")
                try:
                    temp_target.unlink()
                except Exception:
                    pass
                saved_files = [str(wav_target)]
            else:
                # assume already wav
                saved_files = [str(temp_target)]
        voice_ref = _first_wav_in_raw(char_id)
        if voice_ref:
            cur = conn.cursor()
            cur.execute(
                "UPDATE characters SET voice_ref_path = ? WHERE id = ?",
                (voice_ref, char_id),
            )
            conn.commit()
        return {"ok": True, "files": saved_files, "voice_ref_path": voice_ref}
    @app.post("/api/characters/{char_id}/voice_dataset")
    async def upload_voice_dataset_api(char_id: int, file: UploadFile = File(...)):
        return await upload_voice_dataset(char_id, file)

    @app.post("/characters/{char_id}/train_voice")
    async def train_voice(char_id: int):
        raw_dir = ROOT / "data" / "voices" / f"char_{char_id}" / "raw"
        if not raw_dir.exists() or not any(raw_dir.glob("*.wav")):
            raise HTTPException(status_code=400, detail="No WAV files uploaded. Use /voice_dataset first.")
        _set_training_status(char_id, "queued", "")
        thread = threading.Thread(target=_run_training, args=(char_id,), daemon=True)
        thread.start()
        return {"ok": True, "status": "queued"}
    @app.post("/api/characters/{char_id}/train_voice")
    async def train_voice_api(char_id: int):
        return await train_voice(char_id)

    return app


app = create_app()
