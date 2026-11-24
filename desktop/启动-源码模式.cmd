@echo off
rem 忆栈 MemStack —— 源码运行模式（改代码立即生效，无需重新打包）
rem 用 venv 里的 python 直接跑 app.py
cd /d "%~dp0"
start "" "%~dp0.venv\Scripts\pythonw.exe" app.py
