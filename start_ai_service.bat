@echo off
echo Starting DermScan AI Machine Learning API Gateway...
echo.

cd ml_service
call venv\Scripts\activate.bat
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pause
