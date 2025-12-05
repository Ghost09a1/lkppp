Write-Host "⚠️  STOPPING EVERYTHING..." -ForegroundColor Red

# Kill all Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "✅ Killed Python"

# Kill llama-server explicitly
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "✅ Killed llama-server"

# Kill Node.js (Frontend server)
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "✅ Killed Node.js"

# Kill standard "cmd" windows that might wrap processes
# (Dangerous if user has other cmd windows open, but necessary for clean restart)
# Get-Process cmd -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "⏳ Waiting 5 seconds for ports to clear..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "🚀 STARTING MYCANDY LOCAL..." -ForegroundColor Green
& .\start_all.ps1
