@echo off
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 RUN_DIR [ALIAS]
  exit /b 2
)
set ALIAS=%~2
if "%ALIAS%"=="" set ALIAS=best_reliable
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.longrun_tools eval ^
  "%~1" --suite r2 --alias "%ALIAS%"
exit /b %ERRORLEVEL%
