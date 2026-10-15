@echo off
chcp 65001 >nul
echo ======================================
echo    百声AI Strata 大模型服务启动中...
echo ======================================
echo.

:: 获取当前目录（ComfyUI 根目录）
set "ROOT_DIR=%~dp0"

:: Strata 启动脚本路径
set "STRATA_BAT=%ROOT_DIR%ComfyUI\custom_nodes\BSAI-LLM-Strata\Strata\run-q2_0.bat"

if not exist "%STRATA_BAT%" (
    echo 错误：找不到 Strata 启动脚本
    echo 路径: %STRATA_BAT%
    pause
    exit /b 1
)

echo 启动脚本: %STRATA_BAT%
echo.

:: 启动 Strata 服务
cd /d "%ROOT_DIR%ComfyUI\custom_nodes\BSAI-LLM-Strata\Strata"
start "BSAI Strata" cmd /k "%STRATA_BAT"

echo 百声AI Strata 服务已启动！
echo 浏览器访问: http://127.0.0.1:8080
echo.
timeout /t 3 >nul
exit
