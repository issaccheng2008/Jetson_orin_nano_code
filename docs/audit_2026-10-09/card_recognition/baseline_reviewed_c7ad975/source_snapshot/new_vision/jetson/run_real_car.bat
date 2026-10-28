@echo off
cd /d "%~dp0.."
.venv\Scripts\python.exe jetson\run_real_car.py
pause
