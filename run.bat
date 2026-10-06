@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failure
)
".venv\Scripts\python.exe" -m pip install .
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -m thingy serve --open %*
if errorlevel 1 goto failure
exit /b 0
:failure
echo Thingy could not start. Install Python 3.10 or newer and check the message above.
pause
exit /b 1
