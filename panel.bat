@echo off
rem Opens the jobbot control panel. Closing it does not stop the bot.
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 -m jobbot panel
if errorlevel 1 pause
