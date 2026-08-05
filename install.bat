@echo off
setlocal
cd /d "%~dp0"

echo Medical Agent installer
echo =======================

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Please install Python 3.10 or newer and try again.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Installation complete.
echo Run start.bat to open the Medical Agent web app.
