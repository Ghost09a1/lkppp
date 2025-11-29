# Developer Notes

## Project Structure
- **Backend**: FastAPI (`backend/core.py`).
  - Endpoints:
    - `/characters` (CRUD)
    - `/chat/{char_id}` (POST)
    - `/chat_stream/{char_id}` (SSE)
    - `/generate_image` (SD.Next)
    - `/tts` (Local TTS)
    - `/ui` (Static mount)
  - Database: SQLite (`outputs/chat.db`).
  - Models: `backend/models.py` (implied, or inline in core.py).
- **Frontend**:
  - Current `ui/` is a compiled Vite app.
  - Source in `frontend/` (Node/Vite/TS).
  - **Decision**: We will replace `ui/` with a vanilla HTML/CSS/JS app to satisfy "no Node build required" constraint.
- **Launcher**: `app_launcher.py` handles startup of backend, TTS, and browser.

## Integration Points
- **LLM**: `backend/llm.py` handles llama.cpp/Ollama.
- **TTS**: `backend/media.py` calls `backend/tts_server.py`.
- **Images**: `backend/media.py` calls SD.Next API.

## TODOs
1. **Frontend Rewrite**: Build a responsive UI in `ui/` using vanilla JS/CSS.
   - Character Gallery
   - Chat Interface
   - Settings
2. **Backend Polish**:
   - Ensure graceful fallbacks for missing services.
   - Verify `backend/media.py` video stub.
3. **Configuration**:
   - Ensure `settings.json` drives all toggles.

## Missing/Stubbed
- Video generation is stubbed in `backend/media.py`.
- RAG is mentioned but might be basic.
