@echo off
setlocal enableextensions enabledelayedexpansion

REM Change to script directory (project root)
pushd "%~dp0"

REM Ensure entry script exists
if not exist "src\stitch.py" (
  echo Error: entry script `src\stitch.py` not found in project root.
  popd
  exit /b 1
)

REM Venv selection: prefer active venv, otherwise .venv
set "VENV=.venv"
if defined VIRTUAL_ENV (
  set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
) else (
  set "PY=%VENV%\Scripts\python.exe"
)

REM Create venv if missing
if not exist "%PY%" (
  echo Creating virtual environment...
  py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
  if exist "%VENV%\Scripts\python.exe" (
    set "PY=%VENV%\Scripts\python.exe"
  ) else (
    echo Error: failed to create virtual environment.
    popd
    exit /b 1
  )
)

REM Upgrade pip and install dependencies
echo Installing build dependencies...
"%PY%" -m pip install --upgrade pip >nul
if exist "requirements.txt" (
  "%PY%" -m pip install -r requirements.txt
) else (
  "%PY%" -m pip install pyinstaller opencv-contrib-python numpy
)

REM Clean previous PyInstaller artifacts
if exist build  rmdir /s /q build
if exist dist   rmdir /s /q dist
if exist stitch.spec del /q stitch.spec

REM Build single-file exe; bundle test\stitch.ini alongside the exe
echo Building stitch.exe...
"%PY%" -m PyInstaller --noconfirm --clean --onefile --name stitch ^
  --add-data "test\stitch.ini;." ^
  "src\stitch.py"

REM Optional: also bundle test images folder for offline tests
REM Add this to the command above if needed:
REM   --add-data "testdata\img3;testdata\img3"

REM Report result
if exist "dist\stitch.exe" (
  for %%F in ("dist\stitch.exe") do echo Built: %%~fF
  popd
  exit /b 0
) else (
  echo Error: build failed. Check PyInstaller output above.
  popd
  exit /b 1
)
