@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
REM 跑 Eval Harness，检查路由/工具调用准确率
cd /d %~dp0..
python -m eval.eval_harness
pause
