@echo off
title AI Sign Language Translator - Master Launcher
echo =======================================================================
echo   AI SIGN LANGUAGE TRANSLATOR - STARTING APPLICATION
echo =======================================================================
echo.

set PYTHON_CMD=python
where python >nul 2>nul
if %errorlevel% neq 0 (
    if exist "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\python.exe" (
        set PYTHON_CMD="C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\python.exe"
    ) else (
        echo [ERROR] Python not found! Please ensure Python is installed and added to PATH.
        pause
        exit /b 1
    )
)

echo Starting SignTranslate AI Web Application...
%PYTHON_CMD% run.py --web

pause
