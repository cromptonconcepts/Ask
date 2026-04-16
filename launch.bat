@echo off
title TTM Ask Launcher
echo ============================================
echo  Starting TTM Ask Application
echo ============================================
echo.

cd /d "%~dp0"

echo Running portable launcher...
powershell -ExecutionPolicy Bypass -File "%~dp0run_ttm_ask.ps1"

echo.
echo Done! If this was a first run, setup may still be downloading models in the background.
