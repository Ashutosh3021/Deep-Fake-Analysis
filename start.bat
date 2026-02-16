@echo off
echo ==========================================
echo    DeepGuard AI - Starting Server
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Navigate to backend directory
cd /d "%~dp0backend"

REM Check if virtual environment exists, if not create one
if not exist "..\venv" (
    echo Creating virtual environment...
    python -m venv ..\venv
)

REM Activate virtual environment
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r ..\requirements.txt -q

REM Create uploads directory
if not exist "uploads" mkdir uploads

REM Start the server
echo.
echo Starting DeepGuard AI Server...
echo Dashboard will be available at: http://localhost:5000
echo.
echo Press CTRL+C to stop the server
echo.

python api.py

REM Deactivate virtual environment on exit
call ..\venv\Scripts\deactivate.bat