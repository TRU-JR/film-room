@echo off
REM Creates .venv inside this folder and installs every package the app needs into it.
REM Nothing is installed system-wide; pip's download cache also stays inside .venv.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is required. Install it from python.org and run setup.bat again.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
)

set "PIP_CACHE_DIR=%~dp0.venv\pip-cache"
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Start the app with:  run.bat
echo Run the tests with:                  .venv\Scripts\python -m pytest
endlocal
