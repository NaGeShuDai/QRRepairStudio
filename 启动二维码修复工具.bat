@echo off
cd /d "%~dp0"
if exist "dist-v1.6\QRRepairStudio\QRRepairStudio.exe" (
  start "" "dist-v1.6\QRRepairStudio\QRRepairStudio.exe" %*
) else if exist "dist\QRRepairStudio\QRRepairStudio.exe" (
  start "" "dist\QRRepairStudio\QRRepairStudio.exe" %*
) else (
  start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py" %*
)
