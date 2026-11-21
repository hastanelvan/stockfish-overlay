@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python sanal ortami yok. README dosyasindaki kurulum adimlarini uygula.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" auto_app.py
pause
