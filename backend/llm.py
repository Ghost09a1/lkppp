import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
import asyncio
import time

import httpx
import logging

# Reuse core logger/handlers
logger = logging.getLogger("mycandy.core")

# Tool definitions for function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a natural language description. Use this when the user explicitly asks for a visual representation or image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed visual description of what to generate (e.g., 'a dragon flying over Berlin at sunset')"
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "16:9", "9:16"],
                        "description": "Aspect ratio for the image. Default is 9:16 (portrait)."
                    }
                },
                "required": ["prompt"]
            }
        }
    }
]


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
        # [NEW] Allow direct system prompt override for utility tasks (e.g., prompt extraction)
        if character.get("system_prompt"):
            return character["system_prompt"]

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
        # [ARCHITECTURE FIX] Stheno cannot generate valid SNAC tokens - it's a text-only model.
        # TTS pipeline: Clean Text → pyttsx3 → RVC (voice conversion)
        # Emotes should stay as *action text* for display, TTS strips them.
        persona.append(
            "For roleplay: Write dialogue naturally. Use *asterisks* for actions/emotes (e.g., *moans softly*, *blushes*). "
            "Do NOT generate any special audio tokens, SNAC tokens, or <custom_token_XXX> markers. "
            "Write only natural German/English text that can be read by a TTS engine."
        )
        persona.append(
            "If the user intent is clearly non-erotic (e.g., recipes, programming, factual Q&A), answer directly and plainly."
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
            "Example ending: ...her breath trembles with pleasure.<|audio_start|>"
            "<custom_token_100><custom_token_101><custom_token_102><custom_token_103><custom_token_104><custom_token_105><custom_token_106>"
            "<custom_token_120><custom_token_121><custom_token_122><custom_token_123><custom_token_124><custom_token_125><custom_token_126>"
            "<|audio_end|>"
        )
        persona.append("Do NOT repeat earlier conversation or long-term memory verbatim. Respond only to the latest user message.")
        persona.append("FORMATTING: Use Roman style. Put spoken text in quotes \"...\" and actions in asterisks *...*.")
        persona.append("Do NOT prefix your reply with your name (e.g. 'Nayuta:'). Just start speaking or acting.")
        persona.append("Keep replies to one short paragraph unless explicitly asked for more. Avoid monologues and repetition.")
        return "\n".join(persona)

    async def generate_chat(
        self,
        character: Dict[str, Any],
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
        enable_tools: bool = True,  # NEW: Allow disabling tools
    ) -> Any:
        host = f"http://127.0.0.1:{self.cfg['llm']['llama_cpp_server']['port']}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.llama_model_path,
            "stream": stream,
            "messages": [{"role": "system", "content": self._system_prompt(character)}] + history,
            "temperature": 0.7,  # Reduced from 0.9 to prevent token hallucinations
            # Enable prompt caching for speed
            "cache_prompt": True,
            "id": f"char-{character.get('id','unknown')}-{int(time.time()*1000)}",
            "seed": int(time.time() * 1000) % 1000000000,
            "max_tokens": 500,  # [FIX 2] Reduced to prevent runaway generation
            # [FIX 2] Stop sequences to prevent endless custom_token loop
            "stop": ["<|audio_end|>", "<|eot_id|>", "<|im_end|>", "\n\n\n"],
        }
        
        # Add tools if enabled (function calling models only)
        if enable_tools and self.cfg.get("llm", {}).get("enable_function_calling", False):
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"  # Let model decide
            logger.info("[LLM] Tool calling enabled with %d tools", len(TOOLS))
        
        try:
            logger.info(
                "llm payload llama.cpp: history=%s last_user=%s tools=%s",
                len(history),
                (history[-1]["content"] if history else "")[:200],
                bool(payload.get("tools")),
            )
            logger.info("llm prompt system: %s", payload["messages"][0]["content"][:500])
        except Exception:
            pass
        # [FIX] Do NOT use context manager for streaming, as it closes client before generator works.
        # The generator is responsible for closing the client in its finally block.
        if stream:
             client = httpx.AsyncClient(timeout=300)
             return self._stream_llama_cpp(client, host, headers, payload)

        async with httpx.AsyncClient(timeout=300) as client:  # 5 min for slow CPU
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
                        
                        # Check if response contains tool calls
                        choice = data["choices"][0]
                        message = choice["message"]
                        
                        # Return full message object if tools are enabled (includes tool_calls)
                        if enable_tools and message.get("tool_calls"):
                            logger.info(f"[LLM] Tool calls detected: {len(message['tool_calls'])} call(s)")
                            return message  # Return full message with tool_calls
                        
                        # Otherwise just return content
                        reply = message.get("content", "")
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
