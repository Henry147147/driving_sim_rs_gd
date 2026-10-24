@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" dashboard\app.py --repo "%CD%" %*
) else (
  python dashboard\app.py --repo "%CD%" %*
)
if errorlevel 1 pause
