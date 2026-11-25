@echo off
chcp 65001 >nul
cd /d "%~dp0"

"C:\Users\86186\.workbuddy\binaries\python\envs\default\Scripts\python.exe" -m agent.main

echo.
pause
