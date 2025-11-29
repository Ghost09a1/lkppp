# MyCandyLocal - User Guide for Beginners

This guide will help you set up and use MyCandyLocal, even if you're not technical. Follow these steps carefully.

## What You'll Need

1. **A Windows PC** (Windows 10 or 11)
2. **Python 3.10 or newer** ([Download here](https://www.python.org/downloads/))
3. **An AI model file** (GGUF format) - You can find these on Hugging Face
4. **About 30GB of free disk space** (for models and data)

## Step 1: Install Python

1. Download Python from python.org
2. **IMPORTANT**: When installing, check the box that says "Add Python to PATH"
3. Click "Install Now"
4. Wait for installation to complete
5. Open Command Prompt (search "cmd" in Windows) and type: `python --version`
   - You should see something like "Python 3.11.9"

## Step 2: Get MyCandyLocal

1. Download or clone the MyCandyLocal folder to your Desktop
2. You should have a folder at: `C:\Users\YOUR_NAME\Desktop\MyCandyLocal`

## Step 3: Install Required Software

1. Open PowerShell in the MyCandyLocal folder:
   - Right-click on the MyCandyLocal folder
   - Choose "Open in Terminal" or "Open PowerShell window here"

2. Type these commands one at a time:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

3. Wait for everything to download and install (this takes 5-15 minutes)

## Step 4: Get an AI Model

You need an AI model file (GGUF format) for the chatbot to work.

### Option A: llama.cpp (Recommended for beginners)

1. Download a model from Hugging Face (search for "GGUF" models)
   - Example: Look for "MN-12B" or similar uncensored models
   - Download the `.gguf` file (usually 4-8 GB)

2. Put the model file in: `MyCandyLocal\models\llm\`
   - Rename it to something simple like `model.gguf`

3. Download llama.cpp server:
   - Go to https://github.com/ggerganov/llama.cpp/releases
   - Download `llama-b[NUMBER]-bin-win-[YOUR_CPU]-x64.zip`
   - Extract `llama-server.exe` to `MyCandyLocal\bins\`

4. Update the config file:
   - Open `config\settings.json` in Notepad
   - Find the line with `"model_path"`
   - Make sure it says: `"model_path": "models/llm/model.gguf"`

### Option B: Ollama (Easier but less flexible)

1. Download and install Ollama from https://ollama.ai
2. Open PowerShell and type: `ollama pull [model-name]`
   - Example: `ollama pull llama2`
3. Edit `config\settings.json`:
   - Change `"mode": "gguf"` to `"mode": "ollama"`
   - Change `"model"` to match what you downloaded

## Step 5: Optional - Image Generation

If you want AI-generated images:

1. Download and install SD.Next or ComfyUI
2. Start it on port 7860 (SD.Next) or 8002 (ComfyUI)
3. Edit `config\settings.json`:
   - Change `"sdnext_enabled": false` to `"sdnext_enabled": true"`
   - OR change `"comfy_enabled": false` to `"comfy_enabled": true"`

## Step 6: Start the App

### If using llama.cpp:

1. Open PowerShell in MyCandyLocal folder
2. Start the AI server:
   ```powershell
   cd bins
   .\llama-server.exe -m ..\models\llm\model.gguf -c 8192 --port 8081
   ```
   - Leave this window open!

3. Open a NEW PowerShell window in MyCandyLocal folder
4. Type:
   ```powershell
   python app_launcher.py
   ```

5. Your browser should open automatically to the app

### If using Ollama:

1. Make sure Ollama is running (it runs in the background)
2. Open PowerShell in MyCandyLocal folder
3. Type:
   ```powershell
   python app_launcher.py
   ```

4. Your browser should open automatically

## Using the App

### Create Your First Character

1. You'll see the "Characters" tab open
2. Click the "+ New Character" button (pink button in the top right)
3. Fill in the form:
   - **Name**: Give your character a name (e.g., "Sarah")
   - **Description**: A short description (e.g., "A friendly AI companion")
   - **Personality**: Describe how they act (e.g., "Warm, playful, flirty")
   - **Backstory**: Optional background story
   - **Language**: Choose the language you want to chat in
4. Click "Save Character"

### Start Chatting

1. Click the "Chat" tab at the top
2. Click on your character's name in the left sidebar
3. Type a message in the box at the bottom
4. Click "Send" or press Enter
5. Wait for the AI to respond (may take 5-30 seconds depending on your PC)

### Generate Images

1. While in a chat, click the "🎨 Image" button
2. Type a description of what you want to see
3. Click "Generate"
4. Wait for the image to appear (this can take 30-120 seconds)
5. **Note**: This only works if you set up SD.Next or ComfyUI

### Settings

1. Click the "Settings" tab
2. You can see which services are running:
   - **Backend**: Should show "● Online" (green)
   - **LLM**: Should show "● Online" if your AI is running
   - **TTS**: Shows if text-to-speech is working

## Troubleshooting

### "Backend offline" error
- Make sure you ran `python app_launcher.py`
- Check that port 8000 is not used by another program

### "LLM unavailable" message
- Make sure llama-server.exe is running (for llama.cpp)
- OR make sure Ollama is running (for Ollama mode)
- Check that the model path in settings.json is correct

### App is very slow
- The AI model is too big for your PC
- Try a smaller model (look for "Q4" or "Q5" quantization)
- Reduce `"n_ctx"` in settings.json to 4096 or 2048

### Images not generating
- Make sure SD.Next or ComfyUI is running
- Check that you enabled it in settings.json
- Verify the port matches (7860 for SD.Next, 8002 for ComfyUI)

### Browser doesn't open
- Manually open your browser
- Go to: `http://127.0.0.1:8000/ui/`

## Tips for Better Results

### For Better Conversations
- Give your character a detailed personality
- Use clear, complete sentences
- The AI remembers recent conversation, but has limits
- Restart the chat if it seems confused

### For Better Images
- Be very specific in your prompts
- Include style descriptors (e.g., "photorealistic", "anime style")
- Use the "Negative Prompt" to exclude unwanted elements
- Start with lower resolution (512x768) for faster results

### Managing Characters
- You can create multiple characters with different personalities
- Click "Edit" on a character card to modify them
- Upload a custom avatar image (100x100 pixels recommended)

## Keeping Your Data

All your conversations are saved in:
- `outputs\chat.db` (SQLite database)

To backup your data:
1. Copy the entire `outputs` folder
2. Store it somewhere safe
3. To restore, copy it back

To start fresh:
1. Close the app
2. Delete `outputs\chat.db`
3. Start the app again

## Getting Help

If something doesn't work:
1. Check the `logs` folder for error messages
2. Make sure all required files are in the right folders
3. Try restarting both the AI server and the app
4. Check that your model file is compatible (must be GGUF format)

## Privacy Note

**Everything runs on your computer**:
- No internet connection required (after initial setup)
- Your conversations never leave your PC
- No data is sent to any servers
- You have complete privacy

Enjoy your AI companion! 💝
