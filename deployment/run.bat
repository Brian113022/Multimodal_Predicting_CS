@echo off
setlocal
if defined PYTHON_EXECUTABLE (
    set "WEB_PYTHON=%PYTHON_EXECUTABLE%"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "WEB_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "WEB_PYTHON=python"
)
if "%~1"=="" (
    "%WEB_PYTHON%" "%~dp0deploy.py" run
) else (
    "%WEB_PYTHON%" "%~dp0deploy.py" %*
)
exit /b %errorlevel%
