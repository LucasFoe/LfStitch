@echo off
REM Script to delete all files in ./work and ./work/r
REM WARNING: This operation is irreversible!

echo ========================================
echo Deletion of all files!
echo ========================================

REM Check if directories exist
if not exist "img" (
    echo Error: Directory 'img' not found.
    pause
    exit /b 1
)

if not exist "result" (
    echo Warning: Directory 'result' not found, skipping.
)

echo Deleting files in 'work'...
del /q /f "img\*.*"

echo Deleting files in 'work\r'...
del /q /f "result\*.*"

echo.
echo ========================================
echo Deletion complete.
echo ========================================

