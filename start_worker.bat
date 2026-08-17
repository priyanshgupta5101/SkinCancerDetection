@echo off
echo Starting DermScan AI Async Inference Worker...
echo.

cd ml_service
call venv\Scripts\activate.bat
python worker.py

pause
