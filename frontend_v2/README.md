# MyCandyLocal Frontend v2 - Setup Guide

## QuickStart

### 1. Install Dependencies
```powershell
cd frontend_v2
npm install
```

### 2. Development Mode
```powershell
npm run dev
```
This starts the dev server on `http://localhost:5173` with auto-reload.

### 3. Build for Production
```powershell
npm run build
```
Outputs to `../ui_v2/` directory.

---

## Architecture Overview

### Project Structure
```
frontend_v2/
├── src/
│   ├── components/      # React components
│   │   ├── TopBar.tsx   # Header with logo & status
│   │   ├── ChatPanel.tsx# Message display
│   │   └── Composer.tsx # Input area with STT/Image
│   ├── api/
│   │   └── client.ts    # Backend API integration
│   ├── types/
│   │   └── index.ts     # TypeScript definitions
│   ├── App.tsx          # Main application
│   └── main.tsx         # Entry point
├── index.html
├── vite.config.ts       # Vite configuration
├── tailwind.config.js   # Tailwind CSS config
└── package.json
```

### Key Features Implemented

✅ **Chat Interface**
- Message bubbles (user vs assistant styling)
- Auto-scroll to newest message
- Loading indicators

✅ **Voice Input (STT)**
- Click mic button to start/stop recording
- Automatically transcribes to text field
- Respects character language setting

✅ **TTS Playback**
- Play button on assistant messages
- Uses backend TTS service

✅ **Image Generation**
- Image button opens prompt dialog
- Integrates with ComfyUI via backend

✅ **Service Status**
- Green/red dots for LLM, TTS, Image Gen
- Auto-updates every 30 seconds

✅ **Character Selection**
- Left sidebar with character list
- Switch between characters
- Each character has separate chat

---

## Backend Integration

The frontend connects to these endpoints:

| Feature | Endpoint | Method |
|---------|----------|--------|
| Send Message | `/chat/{char_id}` | POST |
| Get Characters | `/characters` | GET |
| STT | `/stt` | POST |
| TTS | `/tts` | POST |
| Image Gen | `/posts/{char_id}/image` | POST |
| Status Check | `/status` | GET |

### Proxy Configuration
In dev mode, Vite proxies API requests to `localhost:8000`.  
In production, the backend serves the built files from `ui_v2/`.

---

## Updating Backend to Serve v2

Edit `backend/core.py`:

```python
# Find the line that mounts static files (near end of create_app function)
# Change from:
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

# To:
app.mount("/", StaticFiles(directory="ui_v2", html=True), name="ui")
```

Or update `config/settings.json`:
```json
{
  "paths": {
    "ui_dir": "ui_v2"
  }
}
```

---

## Development Workflow

1. **Start Backend**:
   ```powershell
   python app_launcher.py
   ```

2. **Start Frontend Dev Server** (in another terminal):
   ```powershell
   cd frontend_v2
   npm run dev
   ```

3. **Access UI**:
   - Dev mode: `http://localhost:5173`
   - Production: `http://localhost:8000` (after building)

---

## Next Steps (Not Yet Implemented)

See `implementation_plan.md` Phase 2 for:
- Prompt presets
- Character editor UI
- File upload
- Theme customization
- Message search
- Right sidebar tabs

---

## Troubleshooting

**Problem**: "Cannot find module" errors  
**Solution**: Run `npm install` in `frontend_v2/`

**Problem**: API calls fail in dev mode  
**Solution**: Ensure backend is running on port 8000

**Problem**: Build output not appearing  
**Solution**: Check `vite.config.ts` outDir setting

**Problem**: TailwindCSS styles not working  
**Solution**: Ensure `index.css` is imported in `main.tsx`
