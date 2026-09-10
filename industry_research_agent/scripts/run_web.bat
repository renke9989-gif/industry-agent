@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM 启动网页版聊天助手（常驻服务，输入网址打开）
cd /d %~dp0..
echo.
echo ============================================================
echo   行业调研智能体 - 网页版启动中...
echo   启动后请在浏览器打开: http://localhost:8000
echo   停止服务请按 Ctrl+C
echo ============================================================
echo.
if exist .venv\Scripts\python.exe (.venv\Scripts\python.exe web_server.py) else (python web_server.py)
pause
