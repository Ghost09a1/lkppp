import base64
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio

import httpx


class MediaRouter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Lazy-initialized STT backends so we don't reload the model per request
        self._fw_model = None
        self._fw_lock = asyncio.Lock()
        self._whisper = None

    async def tts(self, text: str, character: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.config["media"].get("tts_enabled"):
            return {"ok": False, "error": "TTS disabled or no model installed."}
        url = f"http://127.0.0.1:{self.config['media']['tts_port']}/tts"
        payload = {
            "text": text,
            "voice_style": (character or {}).get("voice_style") or "breathy-female",
            "pitch_shift": (character or {}).get("voice_pitch_shift", 0.0),
            "speed": (character or {}).get("voice_speed", 1.0),
            "voice_ref_path": (character or {}).get("voice_ref_path", ""),
            "voice_model_path": (character or {}).get("voice_model_path", ""),
        }
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return {"ok": True, **resp.json()}
            except Exception as exc:
                return {"ok": False, "error": f"TTS service unavailable: {exc}"}

    async def generate_image(
        self,
        prompt: str,
        negative: str = "",
        steps: int = 20,
        width: int = 512,
        height: int = 768,
    ) -> Dict[str, Any]:
        if not self.config["media"].get("sdnext_enabled"):
            return {"ok": False, "error": "SD.Next disabled. Enable in config."}
        url = f"{self.config['media']['sdnext_host']}/sdapi/v1/txt2img"
        payload = {
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": steps,
            "width": width,
            "height": height,
            "sampler_name": "Euler a",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                images = data.get("images") or []
                if not images:
                    return {"ok": False, "error": "No images returned."}
                return {"ok": True, "images_base64": images}
            except Exception as exc:
                return {"ok": False, "error": f"Image service unavailable: {exc}"}

    async def generate_video(self, prompt: str) -> Dict[str, Any]:
        if not self.config["media"].get("video_enabled"):
            return {"ok": False, "error": "Video worker disabled in config."}
        # Placeholder hook for future Wan2.2 integration
        return {"ok": False, "error": "Video generation stub. Enable worker implementation."}

    async def _ensure_fw_model(self):
        """
        Load faster-whisper once (size or path from config).
        """
        if self._fw_model is not None:
            return self._fw_model
        async with self._fw_lock:
            if self._fw_model is not None:
                return self._fw_model
            import logging

            logger = logging.getLogger("mycandy.core")
            try:
                from faster_whisper import WhisperModel  # type: ignore

                media_cfg = self.config.get("media", {})
                model_path = (media_cfg.get("whisper_model_path") or "").strip()
                model_size = media_cfg.get("whisper_model", "small")
                model_arg = model_size
                if model_path:
                    p = Path(model_path)
                    if p.exists():
                        model_arg = str(p)
                    else:
                        logger.warning("whisper_model_path does not exist: %s (falling back to size=%s)", model_path, model_size)
                self._fw_model = WhisperModel(model_arg, device="cpu", compute_type="int8")
                logger.info("loaded faster-whisper model=%s", model_arg)
            except Exception as exc:
                logger.warning("failed to load faster-whisper: %s", exc)
                self._fw_model = None
        return self._fw_model

    def _ensure_whisper(self):
        """
        Fallback to openai/whisper only once.
        """
        if self._whisper is not None:
            return self._whisper
        try:
            import whisper  # type: ignore

            self._whisper = whisper
        except Exception:
            self._whisper = None
        return self._whisper

    async def transcribe_audio(self, audio_path: Path | str, language: str | None = None) -> Optional[str]:
        """
        Best-effort transcription. Tries faster-whisper (cached) or whisper; returns None on failure.
        """
        import logging
        import soundfile as sf
        import numpy as np

        logger = logging.getLogger("mycandy.core")
        lang_cfg = (self.config.get("media", {}).get("whisper_language") or "").strip().lower()
        target_lang = (language or "").strip()
        if not target_lang:
            target_lang = "" if lang_cfg in ("", "auto") else lang_cfg
        if target_lang in ("", "auto"):
            target_lang = None

        # Early silence check to avoid wasting STT cycles
        try:
            data, sr = sf.read(str(audio_path))
            if data.size == 0:
                logger.warning("stt input empty samples file=%s", audio_path)
                return None
            peak = float(np.max(np.abs(data)))
            rms = float(np.sqrt(np.mean(np.square(data))))
            if peak < 1e-4:
                logger.warning("stt input near silent (peak=%s rms=%s) file=%s", peak, rms, audio_path)
                return None
        except Exception as exc:
            logger.warning("stt inspect failed for %s: %s", audio_path, exc)

        # faster-whisper (preferred)
        fw_model = await self._ensure_fw_model()
        if fw_model is not None:
            try:
                segments, _ = fw_model.transcribe(
                    str(audio_path),
                    beam_size=5,
                    temperature=0.0,
                    language=target_lang,
                    vad_filter=True,
                )
                text = "".join(seg.text for seg in segments).strip()
                if text:
                    logger.info(
                        "stt faster-whisper ok len=%s lang=%s file=%s",
                        len(text),
                        target_lang or "",
                        audio_path,
                    )
                    return text
                logger.warning("stt faster-whisper empty result file=%s", audio_path)
            except Exception as exc:
                logger.warning("stt faster-whisper failed: %s", exc)

        # whisper fallback
        whisper_mod = self._ensure_whisper()
        if whisper_mod is not None:
            try:
                model = whisper_mod.load_model("small")
                result = model.transcribe(str(audio_path), language=target_lang)
                text = (result.get("text") or "").strip()
                if text:
                    logger.info(
                        "stt whisper ok len=%s lang=%s file=%s",
                        len(text),
                        target_lang or "",
                        audio_path,
                    )
                    return text
                logger.warning("stt whisper empty result file=%s", audio_path)
            except Exception as exc:
                logger.warning("stt whisper failed: %s", exc)

        return None
