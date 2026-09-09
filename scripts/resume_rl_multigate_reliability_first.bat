@echo off
setlocal
if "%~2"=="" (
  echo Usage: %~nx0 RUN_DIR ADDITIONAL_TIMESTEPS
  exit /b 2
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_rl_multigate_reliability_first.ps1" ^
  -ResumeRunDir "%~1" -AdditionalTimesteps %~2
exit /b %ERRORLEVEL%
