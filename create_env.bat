@echo off
SET CurrentDir="%~dp0"
ECHO %CurrentDir%
echo Python versions installed:
cmd /c pyenv versions
REM Prompt the user to enter the desired Python version
set /p version="Enter the desired Python version: "

REM Create the virtual environment with the selected version
pyenv local %version% && python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip && pip install -r requirements.txt
echo Virtual environment created with Python %version%.
echo.
echo Press Enter to exit...
set /p "=">nul