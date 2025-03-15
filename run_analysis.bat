@echo off
echo IBM Stock Analysis: ARIMA + GARCH Forecasting, EDA, and HTML Report Generation
echo ==============================================================================

REM Step 1: Create virtual environment explicitly at project folder
IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment explicitly
echo Activating virtual environment...
call venv\Scripts\activate

REM Install libraries explicitly
echo Installing required libraries...
pip install --upgrade pip
pip install -r requirements.txt

REM Run analysis script explicitly (generates plots and forecasts)
echo Running analysis script...
python ibm_arima_garch_analysis.py

REM Generate HTML report explicitly
echo Generating HTML report...
python generate_html_report.py

REM Serve the HTML report
echo Starting web server to serve the HTML report...
echo Press Ctrl+C to stop the server when done.
python serve_report.py 59169

echo.
echo Analysis completed. Your HTML report is ready (open "report.html" explicitly).
pause