@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM 启动 Agent 对话
cd /d %~dp0..
if exist .venv\Scripts\python.exe (.venv\Scripts\python.exe main.py) else (python main.py)
pause
