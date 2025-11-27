import base64
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


class MediaRouter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

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

    async def transcribe_audio(self, audio_path: Path | str, language: str | None = None) -> Optional[str]:
        """
        Best-effort transcription. Tries faster-whisper or whisper; returns None on failure.
        """
        import logging

        logger = logging.getLogger("mycandy.core")
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            WhisperModel = None  # type: ignore

        if WhisperModel is not None:
            try:
                model_size = self.config.get("media", {}).get("whisper_model", "small")
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                segments, _ = model.transcribe(
                    str(audio_path),
                    beam_size=5,
                    temperature=0.0,
                    language=language if language else None,
                )
                text = "".join(seg.text for seg in segments).strip()
                if text:
                    logger.info(
                        "stt faster-whisper ok len=%s lang=%s file=%s",
                        len(text),
                        language or "",
                        audio_path,
                    )
                    return text
                logger.warning("stt faster-whisper empty result file=%s", audio_path)
            except Exception as exc:
                logger.warning("stt faster-whisper failed: %s", exc)
                return None
        try:
            import whisper  # type: ignore

            model = whisper.load_model("small")
            result = model.transcribe(str(audio_path), language=language if language else None)
            text = (result.get("text") or "").strip()
            logger.info(
                "stt whisper ok len=%s lang=%s file=%s",
                len(text),
                language or "",
                audio_path,
            )
            return text
        except Exception as exc:
            logger.warning("stt whisper failed: %s", exc)
            return None
