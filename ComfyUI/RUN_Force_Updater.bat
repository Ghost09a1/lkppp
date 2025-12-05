setlocal
set PATH=%~dp0\venv\Library\bin;%~dp0\venv\Lib\site-packages\torch\lib;%~dp0\venv\Lib\site-packages\intel_extension_for_pytorch\bin;%PATH%;%~dp0\MinGit\cmd;%~dp0\venv\Scripts
set PYTHONPYCACHEPREFIX=%~dp0\pycache
.\venv\Scripts\python.exe -s force_updater.py
endlocal
