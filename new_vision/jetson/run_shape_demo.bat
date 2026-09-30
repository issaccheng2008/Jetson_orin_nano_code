@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=py"

echo ============================================
echo   Shape Card Demo
echo   left  = _binary_selective
echo   right = _card_ink
echo   Q/ESC quit    S save debug images
echo ============================================
echo.

if "%~1"=="" (
  %PY% jetson\shape_demo.py --camera
) else (
  %PY% jetson\shape_demo.py %*
)
pause
