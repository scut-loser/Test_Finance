@echo off
setlocal ENABLEDELAYEDEXPANSION

set PYTHON=python
if not "%1"=="" set PYTHON=%1

echo [1/4] Checking Python...
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
  echo Python not found. Please install Python 3.10+ and add to PATH.
  exit /b 1
)

echo [2/4] Creating venv .venv ...
%PYTHON% -m venv .venv
if errorlevel 1 (
  echo Create venv failed.
  exit /b 1
)

echo [3/4] Activating and installing requirements...
call .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Install requirements failed.
  call deactivate
  exit /b 1
)

if not exist .env (
  echo BACKEND_BASE_URL=http://localhost:8080>.env
  echo # BACKEND_TOKEN=your_jwt_token_here>>.env
)

echo Done.
echo   1) Activate: .\.venv\Scripts\activate.bat
echo   2) Run:      python FrontEnd\main.py

endlocal
