@echo off
title Alpha Edge - Institutional AI Trading Terminal
cd /d "%~dp0"
echo ==========================================================================
echo   Alpha Edge - Institutional AI Trading Terminal
echo ==========================================================================
echo Starting launcher...
where py >nul 2>nul && (py run_app.py) || (python run_app.py)
echo.
pause
