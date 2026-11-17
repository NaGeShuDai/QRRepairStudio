@echo off
cd /d "%~dp0"
if exist "dist\QRRepairStudio\QRRepairStudio.exe" (
  start "" "dist\QRRepairStudio\QRRepairStudio.exe" %*
) else (
  start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py" %*
)
