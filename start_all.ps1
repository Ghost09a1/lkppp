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

# --- Determine Python Executable ---
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
    Write-Host "Using venv Python: $python"
}
else {
    $python = "python"
    Write-Host "Using system Python. This might cause issues if dependencies are not installed globally."
}


# --- 1. Start MyCandyLocal Core (Backend, TTS, UI Opener) ---
# This is started first as it's the central piece.
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
        $llamaArgs = @(
            "-m", "'$modelPath'",
            "--host", "'$($config.backend_host)'",
            "--port", $llmSettings.port,
            "--ctx-size", $llmSettings.n_ctx,
            "--threads", $llmSettings.n_threads,
            "--batch-size", $llmSettings.batch
        )
        if ($llmSettings.gpu_layers -gt 0) {
            $llamaArgs += @("--n-gpu-layers", $llmSettings.gpu_layers)
        }
        
        $llamaCommand = "& '$llamaExe' $($llamaArgs -join ' ')"
        Start-NewPowerShell -Name "LLaMA Server" -Command $llamaCommand

    }
    else {
        Write-Warning "LLaMA server binary not found at '$llamaExe', skipping."
    }
}
else {
    Write-Host "LLM mode is '$($config.llm.mode)', skipping dedicated LLaMA server start."
}


# --- 3. Start ComfyUI Server ---
if ($config.media.comfy_enabled) {
    $comfyDir = Join-Path $root "ComfyUI"
    
    # Check if ComfyUI has its own Python (standalone)
    $comfyPython = Join-Path $comfyDir "python_standalone\python.exe"
    $comfyMain = Join-Path $comfyDir "main.py"
    
    if (Test-Path $comfyDir) {
        Write-Host "Starting ComfyUI..."
        
        # Option 1: Use ComfyUI's own launcher (Preferred for Arc/Special setups)
        $comfyLauncher = Join-Path $comfyDir "RUN_Launcher.bat"
        if (Test-Path $comfyLauncher) {
            Write-Host "Using ComfyUI's launcher: $comfyLauncher"
            # Run the bat file inside the new PowerShell window via cmd /c
            $comfyCommand = "Set-Location '$comfyDir'; cmd /c 'RUN_Launcher.bat'"
            Start-NewPowerShell -Name "ComfyUI" -Command $comfyCommand
        }
        # Option 2: Direct Python invocation if standalone Python exists
        elseif ((Test-Path $comfyPython) -and (Test-Path $comfyMain)) {
            Write-Host "Using ComfyUI standalone Python: $comfyPython"
            $comfyCommand = "Set-Location '$comfyDir'; & '$comfyPython' '$comfyMain' --port 8188"
            Start-NewPowerShell -Name "ComfyUI" -Command $comfyCommand
        }
        # Option 3: Use system/venv Python
        elseif (Test-Path $comfyMain) {
            Write-Host "Using system Python for ComfyUI"
            $comfyCommand = "Set-Location '$comfyDir'; & '$python' '$comfyMain' --port 8188"
            Start-NewPowerShell -Name "ComfyUI" -Command $comfyCommand
        }
        else {
            Write-Warning "ComfyUI installation incomplete. Neither launcher nor main.py found in '$comfyDir'."
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
# Assuming RVC should be started if the directory exists.
$rvcDir = Join-Path $root "rvc_webui"
if (Test-Path $rvcDir) {
    $rvcVenvPython = Join-Path $rvcDir "venv\Scripts\python.exe"
    $rvcScript = Join-Path $rvcDir "infer-web.py"
    
    if (Test-Path $rvcScript) {
        $rvcPython = if (Test-Path $rvcVenvPython) { $rvcVenvPython } else { $python }
        Write-Host "Using Python for RVC: $rvcPython"

        # RVC WebUI needs to be run from its own directory
        # ADDED --listen-port 7867 to avoid conflict
        $rvcCommand = "Set-Location '$rvcDir'; & '$rvcPython' '$rvcScript' --port 7867 --pycmd '$rvcPython' --noautoopen"
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
