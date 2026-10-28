@echo off
cd /d "%~dp0.."
echo ============================================
echo   Jetson Vision Demo
echo   Line + QR + Red Bar
echo   Press ESC to quit
echo ============================================
.venv\Scripts\python jetson\vision_main.py
pause
