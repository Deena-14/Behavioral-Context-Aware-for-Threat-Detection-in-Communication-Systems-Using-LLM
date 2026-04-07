@echo off
REM Multimodal LLM-Based Cybersecurity System - Quick Start (Windows)

cls
echo.
echo ================================================================================
echo   MULTIMODAL LLM-BASED CYBERSECURITY SYSTEM - QUICK START
echo   Context-Aware Threat Detection for Communication Networks
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Initializing system...
echo.

REM Create directories
if not exist "logs" mkdir logs
if not exist "results" mkdir results
if not exist "data" mkdir data

echo Directories created: logs, results, data
echo.

REM Display menu
:menu
cls
echo.
echo ================================================================================
echo                           EXECUTION MENU
echo ================================================================================
echo.
echo 1. Single Analysis (Quick - ~30 seconds)
echo 2. Interactive Mode (Continuous monitoring)
echo 3. Batch Analysis (Multiple iterations)
echo 4. Launch Dashboard (Web interface)
echo 5. Install/Update Dependencies
echo 6. View Logs
echo 7. Exit
echo.
echo ================================================================================
echo.
set /p choice=Enter your choice (1-7): 

if "%choice%"=="1" goto single
if "%choice%"=="2" goto interactive
if "%choice%"=="3" goto batch
if "%choice%"=="4" goto dashboard
if "%choice%"=="5" goto install
if "%choice%"=="6" goto logs
if "%choice%"=="7" goto exit
echo Invalid choice. Please try again.
pause
goto menu

:single
cls
echo.
echo Running Single Threat Analysis...
echo.
python main.py --mode single
pause
goto menu

:interactive
cls
echo.
echo Running Interactive Mode...
echo.
echo Controls:
echo   - Press Enter: Run next analysis
echo   - 'q': Quit
echo   - 's': View statistics
echo.
python main.py --mode interactive
pause
goto menu

:batch
cls
echo.
set /p iterations=Enter number of iterations (default 5): 
if "%iterations%"=="" set iterations=5
python main.py --mode batch --iterations %iterations%
pause
goto menu

:dashboard
cls
echo.
echo Launching Dashboard (web interface)...
echo.
echo Dashboard URL: http://localhost:8501
echo.
REM start Streamlit in separate window so script can continue
start "" python -m streamlit run streamlit_app.py
REM wait a moment then open browser
timeout /t 3 /nobreak >nul
start http://localhost:8501

goto menu

:install
cls
echo.
echo Installing/Updating dependencies...
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Installation complete!
pause
goto menu

:logs
cls
echo.
if exist "logs\system.log" (
    echo Displaying system logs (last 50 lines):
    echo.
    powershell -Command "Get-Content 'logs\system.log' -Tail 50"
) else (
    echo No logs found. Run an analysis first.
)
echo.
pause
goto menu

:exit
cls
echo.
echo Thank you for using the Multimodal Cybersecurity System!
echo.
pause
exit /b 0
