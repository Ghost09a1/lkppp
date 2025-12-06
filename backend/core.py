import base64
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
import re
import uuid
import httpx

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
    - strip ALL audio markers and special tokens
    - strip ALL content within asterisks (actions/emotes) unless they contain specific audio tokens we want (which we don't for now)
    - strip generation tags
    """
    import re

    # 1. Remove all XML-like tags <...> (includes <|audio_start|>, <custom_token_x>, etc.)
    # This prevents the TTS from reading "audio start" or "custom token"
    # Note: If we eventually WANT the TTS to play sound effects from tokens, we would selectively keep them.
    # For now, the user says it reads "special tokens", so we nuke them all.
    text = re.sub(r"<[^>]+>", "", text)
    
    # 2. Remove [GENERATE_...] tags and [EMOTE:...]
    text = re.sub(r"\[GENERATE_[^\]]+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[EMOTE:[^\]]+\]", "", text, flags=re.IGNORECASE)

    # Capture the text state before stripping actions for fallback ratio calculation
    without_markers = text

    # 3. Remove all content between asterisks *...* (actions)
    # This is the standard "roleplay action" format. TTS should only read spoken dialogue.
    # We replace it with a pause (comma) to keep cadence natural.
    text = re.sub(r"\*[^*]+\*", ", ", text)

    # 4. Remove residual asterisks
    text = text.replace("*", "")

    # 5. Clean up whitespace
    text = re.sub(r"\s{2,}", " ", text)
    
    # [FALLBACK] Logic for mixed/action messages:
    # If cleaning resulted in empty text OR removed significant portion (>80%) of text (implying mostly actions),
    # switch to "Narration Mode": strip asterisks and read everything.
    # This fixes issues where "Action1 Action2" becomes silent.
    original_len = len(without_markers)
    cleaned_len = len(text)
    is_mostly_action = original_len > 20 and (cleaned_len / original_len) < 0.2
    
    if (not text and re.search(r"[a-zA-Z0-9]", without_markers)) or is_mostly_action:
        # Fallback: Strip asterisks and just read it
        text = without_markers.replace("*", " ").strip()
        # Clean up double spaces from asterisk removal
        text = re.sub(r"\s{2,}", " ", text)

    return text


def _clean_display_text(text: str) -> str:
    """
    Remove ALL custom tokens and markers from text for display in UI.
    This ensures users never see TTS/audio control tokens.
    """
    import re
    
    # [AGGRESSIVE FIX] Remove ALL angle-bracket content <...> including:
    # - <|audio_start|>, <|audio_end|>
    # - <custom_token_XXX>
    # - <|im_start|>, <|im_end|>
    # - Any other <...> tags
    text = re.sub(r"<[^>]+>", "", text)
    
    # Remove [EMOTE:...] and [GENERATE_...] tags
    text = re.sub(r"\[EMOTE:[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[GENERATE_[^\]]*\]", "", text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    
    return text.strip()

def _is_non_erotic_intent(text: str) -> bool:
    """
    Heuristic: treat obvious non-erotic requests (recipes, code, facts) differently.
    """
    t = (text or "").lower()
    keywords = [
        "rezept",
        "keks",
        "cookie",
        "recipe",
        "kochen",
        "backen",
        "programmieren",
        "code",
        "how to",
        "hilfe bei",
        "anleitung",
        "diy",
        "gebrauchsanweisung",
        "fakten",
        "erkläre",
        "erklaere",
        "translate",
    ]
    return any(k in t for k in keywords)


def _has_image_intent(text: str) -> bool:
    """
    Check if the user is explicitly asking for an image.
    """
    t = (text or "").lower()
    phrases = [
        "schick mir ein bild", "zeig mir ein bild", "mach ein bild", "send me a picture", "show me a picture",
        "generate an image", "create an image", "draw me", "male mir", "zeichne mir",
        "schick mir ein foto", "send me a photo", "show me a photo", "selfie",
        "zeig mir deine", "zeig mir dich", "zeig dich", "wie siehst du aus", "what do you look like",
        "schick mir nudes", "send nudes", "zeig mir alles", "show me everything"
    ]
    return any(p in t for p in phrases)


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


def _prune_uploads_dir(upload_dir: Path, keep: int = 1) -> None:
    """
    Keep only the newest `keep` files in the uploads directory to avoid disk growth.
    """
    try:
        files = sorted(
            [p for p in upload_dir.glob("*") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[keep:]:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception:
        pass


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
    negative_prompt: str = ""


class ChatPayload(BaseModel):
    message: str
    enable_tts: bool = True  # Default to True for backward compatibility
    enable_image: bool = False
    force_image: bool = False


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
    prompt: str = ""
    use_last_emote: bool = True
    duration: int = 60


def create_app() -> FastAPI:
    config = load_config()
    db_path = str((ROOT / config["paths"]["db_path"]).resolve())
    db.init_db(db_path)
    conn = db.connect(db_path)
    avatars_dir = ROOT / "outputs" / "avatars"
    images_dir = ROOT / "outputs" / "images"
    videos_dir = ROOT / "outputs" / "videos"
    avatars_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
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
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")
        # Also mount assets at root /assets because index.html expects them there
        assets_dir = ui_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    
    # Mount outputs directory for avatars, images, videos
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

    # Mount reference images directory
    ref_dir = ROOT / "outputs" / "ref_images"
    ref_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/ref_images", StaticFiles(directory=str(ref_dir)), name="ref_images")

    # [FIX 3] Mount avatars directory
    app.mount("/avatars", StaticFiles(directory=str(avatars_dir)), name="avatars")

    @app.get("/")
    async def root():
        return RedirectResponse(url="/ui")


    def _save_image_file(img_b64: str) -> str:
        content = img_b64.split(",", 1)[-1]
        filename = f"img_{uuid.uuid4().hex}.png"
        out_path = images_dir / filename
        out_path.write_bytes(base64.b64decode(content))
        return filename

    def _last_emote(char_id: int) -> str:
        """
        Return the most recent *emote* snippet from stored messages for a character.
        """
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT content FROM messages WHERE character_id = ? ORDER BY id DESC LIMIT 40",
            (char_id,),
        ).fetchall()
        for row in rows:
            text = (row["content"] or "").strip()
            matches = list(re.finditer(r"\*(.+?)\*", text))
            for match in reversed(matches):
                candidate = match.group(1).strip()
                if candidate:
                    return candidate
        return ""

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
        chars = db.list_characters(conn)
        for char in chars:
            refs = db.get_character_reference_images(conn, char["id"])
            char["reference_images"] = [
                {"id": r["id"], "url": f"/ref_images/{Path(r['image_path']).name}"}
                for r in refs
            ]
        return {"characters": chars}
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
        
        # Add reference images
        refs = db.get_character_reference_images(conn, char_id)
        # Convert to list of paths/urls
        character["reference_images"] = [
            {"id": r["id"], "url": f"/ref_images/{Path(r['image_path']).name}"}
            for r in refs
        ]
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
                language=:language, negative_prompt=:negative_prompt
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
        _prune_uploads_dir(uploads_dir, keep=1)

        stt_input = upload_path
        # Normalize audio if ffmpeg available
        normalized = _normalize_audio(upload_path)
        if normalized:
            stt_input = normalized

        text = await llm_client.stt(stt_input, language)
        return {"text": text}
    @app.post("/api/stt")
    async def stt_api(file: UploadFile = File(...), language: str = Form("")):
        return await stt(file, language)

    async def chat(char_id: int, payload: ChatPayload | None, message: str, audio: UploadFile | None):
        character = db.get_character(conn, char_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")

        user_text = ""
        if payload:
            user_text = payload.message
        elif message:
            user_text = message
        
        # Audio input (STT) overrides text if present
        if audio:
            uploads_dir = ROOT / "outputs" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            upload_path = uploads_dir / f"{int(time.time())}.wav"
            with upload_path.open("wb") as f:
                f.write(await audio.read())
            _prune_uploads_dir(uploads_dir, keep=1)
            
            stt_res = await llm_client.stt(upload_path, character.get("language","en"))
            if stt_res:
                user_text = stt_res
            else:
                user_text = "(Audio unintelligible)"
        
        # Only check intent if we have text
        should_gen_image = False
        if user_text:
            should_gen_image = _has_image_intent(user_text) or (payload and payload.force_image)

        # 2. Add user message to DB
        memory.add_message(char_id, "user", user_text)

        # 3. Get context & generate reply
        history = memory.get_context(char_id)
        
        # Add user message to history provided to LLM
        history.append({"role": "user", "content": user_text})
        
        # Check for non-erotic intent to temporarily disable SNAC tokens in the prompt
        # (This logic is handled inside LLMClient._system_prompt now via 'force_neutral' flag if needed,
        # but here we just pass the history.)
        
        # 6. Construct SSE response
        async def event_generator():
            full_reply_text = ""
            logger.info(f"[STREAM] Starting True Streaming for char {char_id}")
            
            try:
                # 1. Stream Text Tokens
                stream_gen = await llm_client.generate_chat(character, history, stream=True)
                
                async for chunk in stream_gen:
                    if chunk:
                        full_reply_text += chunk
                        # [FIX 1] Clean tokens before sending to UI
                        clean_chunk = _clean_display_text(chunk)
                        if clean_chunk.strip():
                            yield {"data": json.dumps({'type': 'token', 'token': clean_chunk})}

                logger.info(f"[STREAM] Text stream done. Length: {len(full_reply_text)}")
                
                # 2. Save to Memory (now that we have full text)
                clean_reply = _dedupe_reply(full_reply_text) # Dedupe logic from original code
                memory.add_message(char_id, "assistant", clean_reply)
                await memory.maybe_summarize(char_id)

                # 3. Parallel Generation: TTS & Image
                
                # --- TTS Generation ---
                tts_text = _clean_tts_text(clean_reply)
                # DEBUG: Log TTS conditions
                logger.info(f"[DEBUG-TTS] payload={payload}, enable_tts={payload.enable_tts if payload else 'N/A'}, tts_text_len={len(tts_text) if tts_text else 0}")
                if (payload and payload.enable_tts) and tts_text:
                    logger.info(f"[TTS] Calling TTS server with text len={len(tts_text)}")
                    try:
                        tts_res = await media.tts(tts_text, character)
                        if tts_res.get("ok") and tts_res.get("audio_base64"):
                            logger.info(f"[TTS] SUCCESS - Got audio_base64 len={len(tts_res['audio_base64'])}")
                            yield {"data": json.dumps({'type': 'audio', 'audio': tts_res["audio_base64"]})}
                        else:
                            logger.warning(f"TTS failed or empty: {tts_res.get('error')}")
                    except Exception as e:
                        logger.error(f"TTS processing failed: {e}")
                else:
                    logger.info(f"[DEBUG-TTS] SKIPPED - payload={payload is not None}, enable_tts={payload.enable_tts if payload else False}, tts_text={bool(tts_text)}")

                # --- Image Generation Logic ---
                should_gen_image = payload and (payload.enable_image or payload.force_image)
                logger.info(f"[DEBUG-IMG] should_gen_image={should_gen_image}, GENERATE_IMAGE_in_reply={'[GENERATE_IMAGE]' in clean_reply.upper()}")
                if should_gen_image or "[GENERATE_IMAGE]" in clean_reply.upper():
                    try:
                        llm_prompt = None
                        
                        if clean_reply and "[GENERATE_IMAGE]" in clean_reply.upper():
                            import re
                            match = re.search(r'\[GENERATE_IMAGE\](.+?)(?:\[|$)', clean_reply, re.IGNORECASE | re.DOTALL)
                            if match:
                                llm_prompt = match.group(1).strip()
                                logger.info(f"[CHAT] Using LLM-generated image prompt: {llm_prompt[:100]}...")
                        
                        if not llm_prompt and should_gen_image:
                            logger.info(f"[CHAT] No [GENERATE_IMAGE] tag found. Attempting smart prompt extraction...")
                            
                            extraction_prompt = (
                                f"Task: Translate this narrative into a comma-separated list of visual tags (PonyXL format) for character: {character.get('name')}.\n"
                                "Rules: Output ONLY the tags. No filler.\n"
                                f"Input Text:\n{clean_reply[:1500]}"
                            )
                            
                            try:
                                # Quick separate call for prompt extraction
                                extraction_history = [{"role": "user", "content": extraction_prompt}]
                                extracted_tags = await llm_client.generate_chat(
                                    {"name": "Prompt Extractor", "force_neutral": True}, 
                                    extraction_history, 
                                    stream=False
                                )
                                
                                if extracted_tags and isinstance(extracted_tags, str) and len(extracted_tags) > 5:
                                    if ":" in extracted_tags:
                                        extracted_tags = extracted_tags.split(":", 1)[1]
                                    llm_prompt = extracted_tags.strip()
                                    logger.info(f"[CHAT] Smart extraction success: {llm_prompt}")
                                else:
                                    llm_prompt = "solo, looking at viewer"
                            except Exception as ext_exc:
                                logger.warning(f"[CHAT] Smart extraction failed: {ext_exc}")
                                llm_prompt = "solo, looking at viewer"
                        
                        if llm_prompt:
                            pony_tags = "score_9, score_8_up, score_7_up"
                            trigger_word = "nayuta (goddess of victory: nikke)"
                            prompt = f"{pony_tags}, {trigger_word}, {character.get('visual_style', '')}, {llm_prompt}"
                            
                            refs = db.get_character_reference_images(conn, char_id)
                            ref_paths = [r["image_path"] for r in refs]
                            neg_prompt = character.get("negative_prompt", "")

                            logger.info(f"[CHAT] Generating image...")
                            img_res = await media.generate_image(prompt, negative=neg_prompt, steps=20, reference_images=ref_paths)
                            
                            if img_res.get("ok") and img_res.get("images_base64"):
                                img_b64_data = img_res.get("images_base64")[0]
                                if img_b64_data and not img_b64_data.startswith("data:"):
                                    img_b64_data = f"data:image/png;base64,{img_b64_data}"
                                
                                filename = _save_image_file(img_b64_data)
                                image_b64_url = f"/outputs/images/{filename}"
                                
                                yield {"data": json.dumps({'type': 'image', 'image': img_b64_data, 'url': image_b64_url})}
                    
                    except Exception as e:
                        logger.error(f"[CHAT] Image generation error: {e}") 
            
            except Exception as e:
                logger.error(f"[STREAM] Critical error in event generator: {e}")
                yield {"data": json.dumps({'type': 'token', 'token': f"\n[System Error: {str(e)}]"})}
            
            yield {"data": "[STREAM_END]"}

        return EventSourceResponse(event_generator())

    # [FIX 4] Real status health checks for LLM and TTS
    @app.get("/status")
    async def status():
        llm_online = False
        tts_online = False
        llm_port = config.get("llm", {}).get("llama_cpp_server", {}).get("port", 8082)
        tts_port = config.get("media", {}).get("tts_port", 8020)
        
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                r = await c.get(f"http://127.0.0.1:{llm_port}/health")
                llm_online = r.status_code == 200
        except Exception:
            pass
        
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                r = await c.get(f"http://127.0.0.1:{tts_port}/health")
                tts_online = r.status_code == 200
        except Exception:
            pass
        
        return {"status": "ok", "backend": "running", "llm_online": llm_online, "tts_online": tts_online}

    # [FIX 3b] Avatar upload endpoint
    @app.post("/characters/{char_id}/avatar")
    async def upload_avatar(char_id: int, file: UploadFile = File(...)):
        character = db.get_character(conn, char_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")
        if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
            raise HTTPException(status_code=400, detail="Only PNG/JPEG/WEBP images are allowed")
        
        # Save avatar file
        prev = character.get("avatar_path") or None
        filename = f"char_{char_id}_{int(time.time())}.png"
        out_path = avatars_dir / filename
        
        # Remove old avatar if present
        if prev:
            old = avatars_dir / prev
            if old.exists():
                try:
                    old.unlink()
                except Exception:
                    pass
        
        data = await file.read()
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(data)).convert("RGBA")
            img = img.resize((100, 100))
            img.save(out_path, format="PNG")
        except Exception:
            out_path.write_bytes(data)
        
        cur = conn.cursor()
        cur.execute("UPDATE characters SET avatar_path = ? WHERE id = ?", (filename, char_id))
        conn.commit()
        return {"ok": True, "avatar_path": filename, "url": f"/avatars/{filename}"}

    # Voice Sample Upload
    @app.post("/characters/{char_id}/voice_sample")
    async def upload_voice_sample(char_id: int, file: UploadFile = File(...)):
        """Upload a voice sample (MP3/WAV) for RVC training."""
        character = db.get_character(conn, char_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")
        
        allowed_types = ("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/m4a", "audio/x-m4a")
        if file.content_type and file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only MP3/WAV/M4A audio files are allowed")
        
        # Create voice data directory
        voice_dir = ROOT / "data" / "voices" / f"char_{char_id}" / "raw"
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with original extension
        ext = Path(file.filename or "sample.mp3").suffix or ".mp3"
        filename = f"sample_{int(time.time())}{ext}"
        out_path = voice_dir / filename
        
        data = await file.read()
        out_path.write_bytes(data)
        
        logger.info(f"[VOICE] Saved voice sample for char {char_id}: {out_path}")
        return {"ok": True, "path": str(out_path), "size": len(data)}

    # Train Voice Model
    @app.post("/characters/{char_id}/train_voice")
    async def train_voice(char_id: int):
        """Start RVC voice training for a character."""
        character = db.get_character(conn, char_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")
        
        _set_training_status(char_id, "queued", "")
        thread = threading.Thread(target=_run_training, args=(char_id,), daemon=True)
        thread.start()
        return {"ok": True, "status": "queued"}

    @app.post("/chat/{char_id}")
    async def chat_endpoint(char_id: int, payload: ChatPayload | None = Body(None), message: str = Form(""), audio: UploadFile | None = File(None)):
        try:
            return await chat(char_id, payload, message, audio)
        except Exception as e:
            logger.error(f"Chat endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()

