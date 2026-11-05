@echo off
setlocal
cd /d "%~dp0"

echo Starting Punjab Tourism Assistant...
echo Open: http://127.0.0.1:8000/
echo Health: http://127.0.0.1:8000/api/health
echo.

python .\chatbot.py
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Chatbot startup or runtime failed with exit code %EXIT_CODE%.
)

echo.
pause
exit /b %EXIT_CODE%
