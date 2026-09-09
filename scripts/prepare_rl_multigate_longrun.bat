@echo off
setlocal
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.longrun_tools prepare ^
  --config configs/rl_multigate_longrun.json %*
exit /b %ERRORLEVEL%
