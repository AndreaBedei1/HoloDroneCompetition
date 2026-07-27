@echo off
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 RUN_DIR [r2^|official]
  exit /b 2
)
set SUITE=%~2
if "%SUITE%"=="" set SUITE=r2
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.longrun_tools eval ^
  "%~1" --suite "%SUITE%"
exit /b %ERRORLEVEL%
