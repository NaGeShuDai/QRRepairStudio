@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Please install dependencies with uv first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py" %*
