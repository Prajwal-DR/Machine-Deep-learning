@echo off
setlocal EnableDelayedExpansion

REM Change to script directory
cd /d "%~dp0"

:MENU
cls
echo ===============================
echo       Claude Switcher
echo ===============================
echo.

set "COUNT=0"
for %%F in (providers\*.env) do (
    set /a COUNT+=1
    set "PROVIDER!COUNT!=%%~nF"
    echo !COUNT!. %%~nF
)

if !COUNT!==0 (
    echo No provider .env files found in "providers\" folder.
    pause
    exit /b 1
)

echo.
set /p "CHOICE=Choose Provider (1-!COUNT!): "

REM Reject empty or non-numeric input before dereferencing PROVIDER<n>.
set "VALID=1"
if not defined CHOICE set "VALID=0"
for /f "delims=0123456789" %%D in ("!CHOICE!") do set "VALID=0"

if "!VALID!"=="1" (
    REM Reject out-of-range numbers.
    if !CHOICE! LSS 1 set "VALID=0"
    if !CHOICE! GTR !COUNT! set "VALID=0"
)

if "!VALID!"=="0" (
    echo.
    echo Invalid choice: !CHOICE!
    echo.
    pause
    goto MENU
)

set "PROVIDER=!PROVIDER%CHOICE%!"

if not defined PROVIDER (
    echo.
    echo Invalid choice: %CHOICE%
    echo.
    pause
    goto MENU
)

echo.
echo Selected Provider : %PROVIDER%
echo.

REM ----------------------------------------------------
REM Delegate everything (dynamic variable discovery,
REM clearing stale vars from this session + the Windows
REM user environment, loading the selected provider, and
REM updating settings.json) to PowerShell. This ensures no
REM variable from a previously selected provider can leak
REM into the newly selected one.
REM ----------------------------------------------------

set "SESSION_VARS_FILE=%TEMP%\claude-switcher-vars-%RANDOM%.tmp"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0switch-provider.ps1" -Provider "%PROVIDER%" -SessionVarsFile "%SESSION_VARS_FILE%"

if errorlevel 1 (
    echo.
    echo Provider switch failed. See errors above.
    pause
    exit /b 1
)

REM Apply the freshly loaded variables to THIS cmd session too
if exist "%SESSION_VARS_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%SESSION_VARS_FILE%") do set "%%A=%%B"
    del "%SESSION_VARS_FILE%" >nul 2>&1
)

echo.
echo ==========================================
echo Active Provider : %PROVIDER%
echo ==========================================
echo.

pause