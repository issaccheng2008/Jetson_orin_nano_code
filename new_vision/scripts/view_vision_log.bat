@echo off
setlocal
where python >nul 2>nul
if errorlevel 1 (
    echo Python 3 is required. Install Python and enable Add Python to PATH.
    pause
    exit /b 1
)
python "%~dp0view_vision_log.py" %*
if errorlevel 1 pause
