@echo off
REM ComfyUI Starter Script for MyCandyLocal
REM This script tries multiple methods to start ComfyUI correctly

cd /d "%~dp0ComfyUI"

echo Starting ComfyUI from: %CD%

REM Method 1: Use RUN_Launcher.bat if it exists
if exist "RUN_Launcher.bat" (
    echo Using RUN_Launcher.bat
    call RUN_Launcher.bat
    goto :end
)

REM Method 2: Use launcher.py with standalone Python
if exist "python_standalone\python.exe" (
    if exist "launcher.py" (
        echo Using standalone Python with launcher.py
        python_standalone\python.exe launcher.py
        goto :end
    )
    if exist "main.py" (
        echo Using standalone Python with main.py
        python_standalone\python.exe main.py --port 8002
        goto :end
    )
)

REM Method 3: Use system Python with main.py
if exist "main.py" (
    echo Using system Python with main.py
    python main.py --port 8002
    goto :end
)

REM If nothing works
echo ERROR: Could not find a way to start ComfyUI
echo Please make sure ComfyUI is properly installed in: %CD%
pause

:end
