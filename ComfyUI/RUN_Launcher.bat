setlocal
set PATH=%~dp0\venv\Library\bin;%~dp0\venv\Lib\site-packages\torch\lib;%~dp0\venv\Lib\site-packages\intel_extension_for_pytorch\bin;%PATH%
set PYTHONPYCACHEPREFIX=%~dp0\pycache
.\venv\Scripts\python.exe -s main.py --port 8188 --disable-smart-memory
endlocal
