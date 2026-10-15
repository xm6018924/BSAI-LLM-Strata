@echo off
chcp 65001 >nul
echo ======================================
echo    BSAI Strata LLM Service Starting...
echo ======================================
echo.

set "ROOT_DIR=%~dp0"
set "STRATA_BAT=%ROOT_DIR%ComfyUI\custom_nodes\BSAI-LLM-Strata\Strata\run-q2_0.bat"

if not exist "%STRATA_BAT%" (
    echo Error: Strata startup script not found
    echo Path: %STRATA_BAT%
    pause
    exit /b 1
)

echo Startup script: %STRATA_BAT%
echo.

cd /d "%ROOT_DIR%ComfyUI\custom_nodes\BSAI-LLM-Strata\Strata"
start "BSAI Strata" cmd /k "%STRATA_BAT"

echo BSAI Strata service started!
echo Browser: http://127.0.0.1:8080
echo.
timeout /t 3 >nul
exit
