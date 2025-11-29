# start_all.ps1
# Starts:
#  - MyCandyLocal (llama.cpp + backend) via app_launcher.py
#  - ComfyUI XPU (Launcher / python)
#  - RVC WebUI (own venv)
# Each in a separate window.

$ErrorActionPreference = "Stop"

# Repo root (folder where this script lives)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Repo root: $root"

# ===== Python for MyCandyLocal (.venv preferred) =====
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonRoot = $venvPython
    Write-Host "Using venv Python for main stack: $pythonRoot"
} else {
    $pythonRoot = "python"
    Write-Host "Using system Python from PATH for main stack."
}

function Start-PythonApp {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$Arguments
    )

    if (-not (Test-Path $WorkingDirectory)) {
        Write-Warning ("{0}: directory '{1}' not found, skipping." -f $Name, $WorkingDirectory)
        return
    }

    if ($PythonExe -ne "python" -and -not (Test-Path $PythonExe)) {
        Write-Warning ("{0}: python executable '{1}' not found, skipping." -f $Name, $PythonExe)
        return
    }

    Write-Host ("Starting {0}..." -f $Name)
    Start-Process -FilePath $PythonExe -WorkingDirectory $WorkingDirectory -ArgumentList $Arguments
}

# -----------------------------------------------------------------------
# 1) LLM server + FastAPI backend (app_launcher.py handles both)
# -----------------------------------------------------------------------
$appLauncher = Join-Path $root "app_launcher.py"
if (Test-Path $appLauncher) {
    Start-PythonApp -Name "MyCandyLocal launcher" `
                    -WorkingDirectory $root `
                    -PythonExe $pythonRoot `
                    -Arguments "app_launcher.py"
} else {
    Write-Warning "app_launcher.py not found in $root - skipping LLM/backend."
}

# -----------------------------------------------------------------------
# 2) ComfyUI XPU (Portable-XPU package in .\ComfyUI)
# -----------------------------------------------------------------------
$comfyDir = Join-Path $root "ComfyUI"
if (Test-Path $comfyDir) {

    # Try: python\python.exe launcher.py (Portable-XPU)
    $comfyPython     = Join-Path $comfyDir "python\python.exe"
    $comfyLauncherPy = Join-Path $comfyDir "launcher.py"

    $launchedComfy = $false

    # Wichtig: -and muss zwischen zwei Ausdruecken stehen
    if ( (Test-Path $comfyPython) -and (Test-Path $comfyLauncherPy) ) {
        Start-PythonApp -Name "ComfyUI XPU (python)" `
                        -WorkingDirectory $comfyDir `
                        -PythonExe $comfyPython `
                        -Arguments "launcher.py"
        $launchedComfy = $true
    } else {
        # Fallback: look for an EXE launcher
        $possibleLaunchers = @(
            "launcher.exe",
            "ComfyUI-Launcher-xpu.exe",
            "ComfyUI-Launcher.exe"
        )

        foreach ($exeName in $possibleLaunchers) {
            $exePath = Join-Path $comfyDir $exeName
            if (Test-Path $exePath) {
                Write-Host ("Starting ComfyUI XPU via {0}..." -f $exeName)
                Start-Process -FilePath $exePath -WorkingDirectory $comfyDir
                $launchedComfy = $true
                break
            }
        }
    }

    if (-not $launchedComfy) {
        Write-Warning "ComfyUI folder found, but no launcher.exe or python\launcher.py detected."
    }
} else {
    Write-Host "ComfyUI folder not found - skipping."
}

# -----------------------------------------------------------------------
# 3) RVC WebUI (has its own venv under rvc_webui\venv)
# -----------------------------------------------------------------------
$rvcDir        = Join-Path $root "rvc_webui"
$rvcScript     = Join-Path $rvcDir "infer-web.py"
$rvcVenvPython = Join-Path $rvcDir "venv\Scripts\python.exe"

if (Test-Path $rvcScript) {
    if (Test-Path $rvcVenvPython) {
        Start-PythonApp -Name "RVC WebUI" `
                        -WorkingDirectory $rvcDir `
                        -PythonExe $rvcVenvPython `
                        -Arguments "infer-web.py"
    } else {
        Start-PythonApp -Name "RVC WebUI (root python)" `
                        -WorkingDirectory $rvcDir `
                        -PythonExe $pythonRoot `
                        -Arguments "infer-web.py"
    }
} else {
    Write-Host "RVC WebUI not found - skipping."
}

Write-Host ""
Write-Host "All launch commands issued."
Write-Host "If the MyCandyLocal UI does not open automatically, open http://127.0.0.1:8000/ in your browser."
