@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM 跑离线 Eval Harness（不调用外部 API）
cd /d %~dp0..
if exist .venv\Scripts\python.exe (.venv\Scripts\python.exe -m eval.eval_harness) else (python -m eval.eval_harness)
pause
