@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [!] .venv not found.
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" -m agent.main

echo.
pause
