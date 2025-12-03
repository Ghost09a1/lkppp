setlocal
set PYTHONPYCACHEPREFIX=%~dp0\pycache
..\ComfyUI_old\venv\Scripts\python.exe -s main.py --port 8188 --disable-smart-memory
endlocal
