@echo off
REM Unified launcher batch file for Selfy Production

REM Get the directory of this script
set "SCRIPT_DIR=%~dp0"

REM Check if the virtual environment exists
if exist "%SCRIPT_DIR%..\selfy_env\Scripts\python.exe" (
    echo Using Python from selfy_env
    "%SCRIPT_DIR%..\selfy_env\Scripts\python.exe" "%SCRIPT_DIR%selfy.py" %*
) else if exist "%SCRIPT_DIR%..\venv\Scripts\python.exe" (
    echo Using Python from venv
    "%SCRIPT_DIR%..\venv\Scripts\python.exe" "%SCRIPT_DIR%selfy.py" %*
) else if exist "%SCRIPT_DIR%..\.venv\Scripts\python.exe" (
    echo Using Python from .venv
    "%SCRIPT_DIR%..\.venv\Scripts\python.exe" "%SCRIPT_DIR%selfy.py" %*
) else if exist "%SCRIPT_DIR%..\env\Scripts\python.exe" (
    echo Using Python from env
    "%SCRIPT_DIR%..\env\Scripts\python.exe" "%SCRIPT_DIR%selfy.py" %*
) else (
    echo WARNING: Could not find virtual environment. Using system Python.
    echo This may cause issues if dependencies are not installed in the system Python.
    python "%SCRIPT_DIR%selfy.py" %*
)

REM Pause at the end to see any error messages
pause
