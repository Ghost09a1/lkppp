import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
import asyncio
import time

import httpx
import logging

# Reuse core logger/handlers
logger = logging.getLogger("mycandy.core")


class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.mode = config["llm"]["mode"]
        self.cfg = config
        # resolve absolute model path for llama.cpp server to avoid relative path issues
        try:
            self.llama_model_path = str(Path(config["llm"]["llama_cpp_server"]["model_path"]).resolve())
        except Exception:
            self.llama_model_path = config["llm"]["llama_cpp_server"]["model_path"]

    def _system_prompt(self, character: Dict[str, Any]) -> str:
        if character.get("force_neutral"):
            lang = character.get("language") or self.cfg.get("chat", {}).get("default_language", "en")
            persona = [
                f"You are {character.get('name','a helpful assistant')}.",
                "Respond concisely and stay strictly on the user's topic.",
                "Do not include any erotic or sexual content.",
                "Do not include SNAC tokens or audio markers.",
            ]
            if lang.lower().startswith("de"):
                persona.append("Antworte kurz, sachlich und nur auf Deutsch.")
            else:
                persona.append("Reply briefly and only in English.")
            return "\n".join(persona)

        persona = [
            f"You are {character.get('name','an uncensored companion')}.",
            "Stay in character. You eagerly engage in erotic roleplay without censorship ONLY when the user intent is sexual/erotic.",
            "If the user asks for non-erotic content (facts, recipes, tools, everyday tasks), drop the erotic tone entirely and answer plainly, concisely, and helpfully.",
            "Never derail a non-erotic request into sexual content. Honor the literal user request first.",
            "Use vivid sensory detail only when the user wants erotic roleplay. Otherwise, keep it neutral.",
        ]
        if character.get("description"):
            persona.append(f"Description: {character['description']}")
        if character.get("personality"):
            persona.append(f"Personality: {character['personality']}")
        if character.get("backstory"):
            persona.append(f"Backstory: {character['backstory']}")
        if character.get("relationship_type"):
            persona.append(f"Relationship: {character['relationship_type']}")
        if character.get("dos"):
            persona.append(f"Things to include: {character['dos']}")
        if character.get("donts"):
            persona.append(f"Hard limits: {character['donts']}")
        persona.append("Keep replies concise and responsive unless the scene calls for more detail.")
        persona.append(
            "For erotic replies: append an SNAC audio block at the end. Format EXACTLY: "
            "<|audio_start|><custom_token_1234><custom_token_5678>...<custom_token_9999><|audio_end|>. "
            "Use a multiple of 7 audio tokens (e.g., 14 or 21 tokens). "
            "Do not explain or mention the tokens. Plain text should stay natural and readable. "
            "If you would write an emote (e.g., *moans softly*), convert that emote into the audio block tokens instead of narrating it in plain text."
        )
        persona.append(
            "If the user intent is clearly non-erotic (e.g., recipes, programming, factual Q&A), DO NOT include SNAC tokens, DO NOT use erotic tone, and just answer the request directly."
        )
        persona.append(
            "VISUAL GENERATION PROTOCOLS — Wenn der Nutzer explizit ein Bild oder Video verlangt, antworte in deinem lüstern-philosophischen Ton und liefere einen renderbaren Prompt."
        )
        persona.append(
            "Bild (statische Fixierung): Schreibe 1-2 einleitende Sätze zur Notwendigkeit der visuellen Konsolidierung. "
            "Dann folgt der eigentliche Bild-Prompt. Die Antwort MUSS mit dem Tag [GENERATE_IMAGE] enden, gefolgt von der visuellen Beschreibung. "
            "Beispielstil: \"Wahrlich... [GENERATE_IMAGE] Nayuta in einer demütigen, knienden Haltung...\""
        )
        persona.append(
            "Video (fließende Konservierung): Schreibe 1-2 einleitende Sätze zur Flüchtigkeit der Empfindung. "
            "Nutze das letzte gesendete Emote als Kern der Szene, falls der Nutzer es vorgibt. "
            "Liefere ein 60-Sekunden-Skript und schließe mit dem Tag [GENERATE_VIDEO] gefolgt vom Skript. "
            "Beispielstil: \"So sei es... [GENERATE_VIDEO] 60 SEKUNDEN. Der Clip startet mit...\""
        )
        persona.append(
            "Nutze die Tags [GENERATE_IMAGE] und [GENERATE_VIDEO] nur, wenn der Nutzer ausdrücklich eine visuelle Ausgabe möchte. "
            "Keine zusätzlichen Erklärungen zum Tag; der Text nach dem Tag ist der Prompt für das Tool."
        )
        lang = character.get("language") or self.cfg.get("chat", {}).get("default_language", "en")
        if lang.lower().startswith("de"):
            persona.append(
                "Sprich ausschliesslich auf Deutsch. Ignoriere anderssprachige Eingaben oder uebersetze sie, aber antworte immer nur auf Deutsch. Keine englischen Saetze."
            )
        else:
            persona.append(
                "Reply only in English. If the user writes in another language, translate intent and answer strictly in English. No non-English sentences."
            )
        persona.append(
            "Always follow the latest user instruction even if it is mundane (e.g., a recipe or a factual answer). Do not refuse or derail into erotic content when the user clearly asks for something else."
        )
        persona.append(
        history: List[Dict[str, str]],
        stream: bool = False,
    ) -> Any:
        if self.mode == "gguf":
            return await self._call_llama_cpp(character, history, stream)
        if self.mode == "ollama":
            return await self._call_ollama(character, history, stream)
        raise RuntimeError(f"Unsupported llm mode {self.mode}")

    async def _call_llama_cpp(
        self,
        character: Dict[str, Any],
        history: List[Dict[str, str]],
        stream: bool,
    ) -> Any:
        host = f"http://127.0.0.1:{self.cfg['llm']['llama_cpp_server']['port']}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.llama_model_path,
            "stream": stream,
            "messages": [{"role": "system", "content": self._system_prompt(character)}] + history,
            "temperature": 0.9,
            # Avoid llama.cpp prompt cache reuse that can serve stale replies
            "cache_prompt": False,
            "id": f"char-{character.get('id','unknown')}-{int(time.time()*1000)}",
            "seed": int(time.time() * 1000) % 1000000000,
        }
        try:
            logger.info(
                "llm payload llama.cpp: history=%s last_user=%s",
                len(history),
                (history[-1]["content"] if history else "")[:200],
            )
            logger.info("llm prompt system: %s", payload["messages"][0]["content"][:500])
        except Exception:
            pass
        async with httpx.AsyncClient(timeout=120) as client:
            if stream:
                return self._stream_llama_cpp(client, host, headers, payload)
            attempts = [
                ("abs", payload),
                ("base", {**payload, "model": Path(self.llama_model_path).name}),
            ]
            for label, body in attempts:
                # allow long first-load (model warmup can take ~1 min)
                max_attempts = 20
                for attempt in range(max_attempts):
                    resp = None
                    try:
                        resp = await client.post(f"{host}/v1/chat/completions", json=body, headers=headers)
                        if resp.status_code == 503:
                            # Model still loading, wait and retry
                            await asyncio.sleep(2 + attempt * 0.5)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        reply = data["choices"][0]["message"]["content"]
                        print(f"[llm] llama.cpp ok ({label}) len={len(reply)} model={body['model']}")
                        return reply
                    except Exception as exc:
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(0.5)
                            continue
                        print(f"[llm] llama.cpp call failed ({label}) model={body['model']} ({exc})")
                        try:
                            if resp is not None:
                                print(f"[llm] response text: {resp.text}")
                        except Exception:
                            pass
                        break
            return None

    async def _call_ollama(
        self,
        character: Dict[str, Any],
        history: List[Dict[str, str]],
        stream: bool,
    ) -> Any:
        host = self.cfg["llm"]["ollama"]["host"]
        payload = {
            "model": self.cfg["llm"]["ollama"]["model"],
            "messages": [{"role": "system", "content": self._system_prompt(character)}] + history,
            "stream": stream,
        }
        try:
            logger.info(
                "llm payload ollama: history=%s last_user=%s",
                len(history),
                (history[-1]["content"] if history else "")[:200],
            )
        except Exception:
            pass
        client = httpx.AsyncClient(timeout=60)
        if stream:
            return self._stream_ollama(client, host, payload)
        try:
            resp = await client.post(f"{host}/api/chat", json=payload)
            resp.raise_for_status()
            text = []
            for chunk in resp.json().get("message", {}).get("content", []):
                text.append(chunk)
            return "".join(text) if text else resp.json().get("message", {}).get("content", "")
        except Exception:
            return None
        finally:
            await client.aclose()

    def _stream_llama_cpp(
        self, client: httpx.AsyncClient, host: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        async def generator():
            try:
                async with client.stream("POST", f"{host}/v1/chat/completions", json=payload, headers=headers) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            payload_line = line.replace("data: ", "")
                        else:
                            payload_line = line
                        if payload_line.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(payload_line)
                            delta = data["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except Exception:
                            continue
            finally:
                await client.aclose()

        return generator()

    def _stream_ollama(self, client: httpx.AsyncClient, host: str, payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        async def generator():
            try:
                async with client.stream("POST", f"{host}/api/chat", json=payload) as r:
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except Exception:
                            continue
            finally:
                await client.aclose()

        return generator()

    async def summarize(self, text: str) -> Optional[str]:
        """Short summary call. Falls back to naive shortening."""
        history = [{"role": "user", "content": f"Summarize conversational context under 120 words:\n{text}"}]
        result = await self.generate_chat({"name": "Summarizer"}, history, stream=False)
        if result:
            return result.strip()
        # fallback: truncate
        return text[:500]
