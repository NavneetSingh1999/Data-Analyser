@echo off
cd /d "%~dp0"
echo ========================================================
echo Installing Stock Market Data Analyser dependencies...
echo ========================================================
where py >nul 2>nul && (py -m pip install -r requirements.txt) || (python -m pip install -r requirements.txt)
echo.
echo ========================================================
echo Installation complete! You can now run run.bat.
echo ========================================================
pause
