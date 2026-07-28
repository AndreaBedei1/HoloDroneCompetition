@echo off
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 RUN_DIR
  exit /b 2
)
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.longrun_tools status "%~1"
exit /b %ERRORLEVEL%
