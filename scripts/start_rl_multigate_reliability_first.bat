@echo off
setlocal
set TOTAL=%~1
if "%TOTAL%"=="" set TOTAL=1000000
set RUN_NAME=%~2
if "%RUN_NAME%"=="" set RUN_NAME=r2_reliability_first_seed23001
set SEED=%~3
if "%SEED%"=="" set SEED=23001
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_rl_multigate_reliability_first.ps1" ^
  -TotalTimesteps %TOTAL% -RunName "%RUN_NAME%" -Seed %SEED%
exit /b %ERRORLEVEL%
