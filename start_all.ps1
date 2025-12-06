# Stop on errors
$ErrorActionPreference = "Stop"

# Get the absolute path to the project's root directory
$root = (Get-Item -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)).FullName
Set-Location $root

Write-Host "Project Root: $root"

# --- Read Configuration ---
$configPath = Join-Path $root "config\settings.json"
if (-not (Test-Path $configPath)) {
    Write-Error "CRITICAL: settings.json not found at '$configPath'"
    exit 1
}
$config = Get-Content $configPath | ConvertFrom-Json
Write-Host "Configuration loaded successfully."

# --- Helper function to start apps in a new window ---
function Start-NewPowerShell {
    param(
        [string]$Name,
        [string]$Command
    )
    $psCommand = "Set-Location '$root'; $Command"
    Write-Host "Starting '$Name' in a new PowerShell window..."
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $psCommand
}

# --- Determine Main Venv Python Executable ---
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
    Write-Host "Using Main venv Python: $python"
}
else {
    $python = "python"
    Write-Host "Using system Python. This might cause issues if dependencies are not installed globally."
}


# --- Helper function to kill process on a specific port ---
function Kill-PortProcess {
    param([int]$Port, [string]$Name)
    $procs = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    if ($procs) {
        Write-Host "Stopping existing '$Name' process(es) on port $Port..."
        foreach ($p in $procs) {
            try { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue } catch {}
        }
        Start-Sleep -Seconds 3
    }
}

# --- 1. Start MyCandyLocal Core (Backend, TTS, UI Opener) ---
Kill-PortProcess -Port 8000 -Name "MyCandyLocal Core"
Start-NewPowerShell -Name "MyCandyLocal Core" -Command "& '$python' app_launcher.py"


# --- 2. Start LLM Server (GGUF/llama.cpp) ---
if ($config.llm.mode -eq "gguf") {
    $llmSettings = $config.llm.llama_cpp_server
    $llamaExe = Join-Path $root $llmSettings.binary_path
    $modelPath = Join-Path $root $llmSettings.model_path

    if (Test-Path $llamaExe) {
        if (-not (Test-Path $modelPath)) {
            Write-Warning "LLM model not found at '$modelPath'. The LLM server will likely fail."
        }
        
        Write-Host "Preparing to start LLaMA server on port $($llmSettings.port)..."
        
        # Enhanced cleanup: Kill by port AND by process name
        Kill-PortProcess -Port $llmSettings.port -Name "LLaMA Server"
        Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force
        Get-Process | Where-Object { $_.Path -like "*llama-server.exe*" } -ErrorAction SilentlyContinue | Stop-Process -Force
        
        Start-Sleep -Seconds 2
        
        $llamaArgs = @(
            "-m", "`"$modelPath`"",
            "--host", "$($config.backend_host)",
            "--port", $llmSettings.port,
            "--ctx-size", $llmSettings.n_ctx,
            "--threads", $llmSettings.n_threads,
            "--batch-size", $llmSettings.batch
        )
        if ($llmSettings.gpu_layers -gt 0) {
            $llamaArgs += @("--n-gpu-layers", $llmSettings.gpu_layers)
        }
        
        $llamaCommand = "& '$llamaExe' $($llamaArgs -join ' ')"
        Write-Host "LLaMA command: $llamaCommand"
        Start-NewPowerShell -Name "LLaMA Server" -Command $llamaCommand
    }
    else {
        Write-Warning "LLaMA server binary not found at '$llamaExe', skipping."
    }
}
else {
    Write-Host "LLM mode is '$($config.llm.mode)', skipping dedicated LLaMA server start."
}

# --- 3. Start ComfyUI Server (Intel Arc Optimized) ---
# WICHTIG: Muss RUN_Launcher.bat nutzen, da es die DLL-Pfade für Intel IPEX setzt!
if ($config.media.comfy_enabled) {
    $comfyDir = Join-Path $root "ComfyUI"
    $comfyLauncher = Join-Path $comfyDir "RUN_Launcher.bat"
    
    if (Test-Path $comfyDir) {
        Write-Host "Starting ComfyUI (Intel Arc Optimized)..."
        Kill-PortProcess -Port 8188 -Name "ComfyUI"
        
        if (Test-Path $comfyLauncher) {
            Write-Host "Using ComfyUI Intel Arc Launcher: $comfyLauncher"
            # The bat file sets critical PATH for Intel IPEX DLLs:
            # - venv\Library\bin
            # - venv\Lib\site-packages\torch\lib
            # - venv\Lib\site-packages\intel_extension_for_pytorch\bin
            $comfyCommand = "Set-Location '$comfyDir'; cmd /c 'RUN_Launcher.bat'"
            Start-NewPowerShell -Name "ComfyUI" -Command $comfyCommand
        }
        else {
            Write-Warning "RUN_Launcher.bat not found. Trying direct Python (may fail without DLL paths)..."
            $comfyVenvPython = Join-Path $comfyDir "venv\Scripts\python.exe"
            $comfyMain = Join-Path $comfyDir "main.py"
            if ((Test-Path $comfyVenvPython) -and (Test-Path $comfyMain)) {
                # Set PATH manually for Intel IPEX DLLs
                $env:PATH = "$comfyDir\venv\Library\bin;$comfyDir\venv\Lib\site-packages\torch\lib;$comfyDir\venv\Lib\site-packages\intel_extension_for_pytorch\bin;$env:PATH"
                $comfyCommand = "Set-Location '$comfyDir'; `$env:PATH = '$comfyDir\venv\Library\bin;$comfyDir\venv\Lib\site-packages\torch\lib;$comfyDir\venv\Lib\site-packages\intel_extension_for_pytorch\bin;' + `$env:PATH; & '$comfyVenvPython' '$comfyMain' --listen --disable-smart-memory"
                Start-NewPowerShell -Name "ComfyUI" -Command $comfyCommand
            }
            else {
                Write-Warning "ComfyUI installation incomplete."
            }
        }
    }
    else {
        Write-Warning "ComfyUI directory not found at '$comfyDir', skipping."
    }
}
else {
    Write-Host "ComfyUI is disabled in settings.json, skipping."
}


# --- 4. Start RVC WebUI Server ---
$rvcDir = Join-Path $root "rvc_webui"
if (Test-Path $rvcDir) {
    $rvcVenvPython = Join-Path $rvcDir "venv\Scripts\python.exe"
    $rvcScript = Join-Path $rvcDir "infer-web.py"
    
    if (Test-Path $rvcScript) {
        $rvcPython = if (Test-Path $rvcVenvPython) { $rvcVenvPython } else { $python }
        Write-Host "Using Python for RVC: $rvcPython"

        Kill-PortProcess -Port 7867 -Name "RVC WebUI"
        # RVC WebUI with --nolaunch to prevent auto-browser
        $rvcCommand = "Set-Location '$rvcDir'; & '$rvcPython' '$rvcScript' --port 7867 --pycmd '$rvcPython' --nolaunch"
        Start-NewPowerShell -Name "RVC WebUI" -Command $rvcCommand
    }
    else {
        Write-Warning "RVC script 'infer-web.py' not found in '$rvcDir', skipping."
    }
}
else {
    Write-Host "RVC directory not found, skipping."
}

Write-Host "`nAll selected servers have been launched in separate windows."
Write-Host "The main 'MyCandyLocal Core' window will open the browser once the backend is ready."
