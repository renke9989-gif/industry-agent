@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM 一行跑预设 demo 对话（自动取 eval_cases.json 第一个用例）
cd /d %~dp0..
python scripts\agent_tools.py demo
pause
