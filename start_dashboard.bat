@echo off
setlocal
cd /d "%~dp0"
title Smart Medical Lighting Dashboard

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found.
    pause
    exit /b 1
)

powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
    echo ERROR: Port 8000 is already in use.
    pause
    exit /b 1
)

start "" /min powershell.exe -NoProfile -Command "$url='http://127.0.0.1:8000'; $deadline=(Get-Date).AddSeconds(120); while ((Get-Date) -lt $deadline) { try { $r=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop; if ($r.StatusCode -eq 200) { Start-Process $url; exit 0 } } catch {} ; Start-Sleep -Milliseconds 500 }; exit 1"

echo Starting Smart Medical Lighting Dashboard...
echo URL: http://127.0.0.1:8000
echo Press Ctrl+C to stop the service.
echo.

".venv\Scripts\python.exe" -m uvicorn web.app:app --host 127.0.0.1 --port 8000

echo.
echo Dashboard service exited.
pause
endlocal