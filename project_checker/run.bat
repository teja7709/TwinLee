@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 (
    py -3 backend\server.py
    goto :done
  )
)

where python >nul 2>nul
if %errorlevel%==0 (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 (
    python backend\server.py
    goto :done
  )
)

echo Python 3.10 or newer is required. Install Python from https://www.python.org/downloads/
echo On Windows, select "Add Python to PATH" during installation.
pause
:done
endlocal

