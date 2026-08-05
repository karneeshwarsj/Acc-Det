@echo off
title AccidentAI — Startup
color 0A

echo.
echo  ================================================
echo     ACCIDENT-DETECTION-APP  ^|  Startup
echo  ================================================
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: ── Check Node ───────────────────────────────────────────────────────────────
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
)

:: ── Backend: create venv if missing ──────────────────────────────────────────
echo  [1/4] Setting up Python virtual environment...
if not exist "backend\venv\Scripts\activate.bat" (
    echo        Creating venv for the first time...
    python -m venv backend\venv
)

:: ── Backend: install deps if needed ──────────────────────────────────────────
echo  [2/4] Checking Python dependencies...
backend\venv\Scripts\pip install -r backend\requirements.txt --quiet --disable-pip-version-check

:: ── Frontend: install deps if needed ─────────────────────────────────────────
echo  [3/4] Checking Node.js dependencies...
if not exist "frontend\node_modules" (
    echo        Installing npm packages for the first time...
    cd frontend
    npm install --silent
    cd ..
)

:: ── Launch both servers ───────────────────────────────────────────────────────
echo  [4/4] Starting backend and frontend...
echo.
echo  Backend API  ^>  http://localhost:8000
echo  Frontend UI  ^>  http://localhost:5173
echo  API Docs     ^>  http://localhost:8000/docs
echo.
echo  Press Ctrl+C in either window to stop.
echo  ================================================
echo.

:: Start backend in a new window
start "AccidentAI — Backend (FastAPI)" cmd /k "cd /d "%~dp0backend" && venv\Scripts\uvicorn main:app --reload --port 8000 --host 0.0.0.0"

:: Give backend a 3-second head start
timeout /t 3 /nobreak >nul

:: Start frontend in a new window
start "AccidentAI — Frontend (React)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

:: Open browser after a short delay
timeout /t 5 /nobreak >nul
start "" "http://localhost:5173"

echo  Both servers are running in separate windows.
echo  This window can be closed.
echo.
pause
