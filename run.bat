@echo off
REM Starts the app with the virtual environment's Python (runs setup first if .venv is missing).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  call setup.bat
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py %*
