import base64
from pathlib import Path
import io
from typing import Any, Dict, Optional
import asyncio
import base64
import subprocess
import uuid
import tempfile

import httpx
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline


class MediaRouter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Lazy-initialized STT backends so we don't reload the model per request
        self._fw_model = None
        self._fw_lock = asyncio.Lock()
        self._whisper = None
        self._sd_pipe: StableDiffusionXLPipeline | None = None
        self._svd_pipe: StableVideoDiffusionPipeline | None = None
        self._sd_lock = asyncio.Lock()
        self._svd_lock = asyncio.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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

    async def _ensure_sdxl(self) -> StableDiffusionXLPipeline | None:
        if self._sd_pipe is not None:
            return self._sd_pipe
        async with self._sd_lock:
            if self._sd_pipe is not None:
                return self._sd_pipe
            model_path = (self.config["media"].get("sdxl_model_path") or "").strip()
            if not model_path or not Path(model_path).exists():
                return None
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=self._torch_dtype,
                )
                pipe.enable_attention_slicing()
                if self._device == "cuda":
                    pipe = pipe.to(self._device)
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
                self._sd_pipe = pipe
                return pipe
            except Exception:
                self._sd_pipe = None
                return None

    async def _ensure_svd(self) -> StableVideoDiffusionPipeline | None:
        if self._svd_pipe is not None:
            return self._svd_pipe
        async with self._svd_lock:
            if self._svd_pipe is not None:
                return self._svd_pipe
            model_id = (self.config["media"].get("svd_model_id") or "").strip() or "stabilityai/stable-video-diffusion-img2vid-xt"
            try:
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self._torch_dtype,
                    variant="fp16" if self._torch_dtype == torch.float16 else None,
                )
                pipe.enable_attention_slicing()
                if self._device == "cuda":
                    pipe = pipe.to(self._device)
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
                self._svd_pipe = pipe
                return pipe
            except Exception:
                self._svd_pipe = None
                return None

    async def generate_image(
        self,
        prompt: str,
        negative: str = "",
        steps: int = 20,
        width: int = 512,
        height: int = 768,
    ) -> Dict[str, Any]:
        # Prefer local diffusers SDXL if available; fall back to SD.Next
        pipe = await self._ensure_sdxl()
        if pipe is not None:
            try:
                generator = None
                seed = self.config["media"].get("image_seed")
                if isinstance(seed, int):
                    generator = torch.Generator(device=self._device).manual_seed(seed)
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    guidance_scale=7.0,
                    generator=generator,
                )
                img = result.images[0]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return {"ok": True, "images_base64": [b64]}
            except Exception as exc:
                return {"ok": False, "error": f"Local SDXL failed: {exc}"}

        if not self.config["media"].get("sdnext_enabled"):
            return {"ok": False, "error": "Image generation unavailable (no SDXL model and SD.Next disabled)." }
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

    async def generate_video(
        self,
        prompt: str,
        duration: int = 60,
        width: int = 960,
        height: int = 540,
    ) -> Dict[str, Any]:
        if not self.config["media"].get("video_enabled"):
            return {"ok": False, "error": "Video worker disabled in config."}
        frame = await self.generate_image(prompt, "", 18, width, height)
        if not frame.get("ok"):
            return {"ok": False, "error": frame.get("error", "Frame generation failed.")}
        images = frame.get("images_base64") or []
        if not images:
            return {"ok": False, "error": "No frame returned for video."}

        raw = images[0]
        b64_content = raw.split(",", 1)[-1]
        try:
            data = base64.b64decode(b64_content)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Invalid frame data: {exc}"}

        root = Path(__file__).resolve().parents[1]
        videos_dir = root / "outputs" / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        still_path = videos_dir / f"frame_{uuid.uuid4().hex}.png"
        video_path = videos_dir / f"clip_{uuid.uuid4().hex}.mp4"
        still_path.write_bytes(data)

        # Create a simple 60s loop from the still frame; keep bitrate small
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(still_path),
                "-t",
                str(max(5, duration)),
                "-vf",
                "scale=1280:-2,format=yuv420p",
                "-r",
                "24",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode != 0 or not video_path.exists():
                return {"ok": False, "error": "ffmpeg failed to render video."}
        except FileNotFoundError:
            return {"ok": False, "error": "ffmpeg not installed; cannot render video."}

        preview = raw if raw.startswith("data:") else f"data:image/png;base64,{raw}"
        return {
            "ok": True,
            "video_path": str(video_path),
            "cover_base64": preview,
        }

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
