@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM 检查所有 Agent 的 System Prompt + Tool 绑定是否一致
cd /d %~dp0..
python scripts\check_agents.py
pause
