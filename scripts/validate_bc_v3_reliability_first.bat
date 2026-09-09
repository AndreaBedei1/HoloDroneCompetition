@echo off
setlocal
set OUT=%~1
if "%OUT%"=="" set OUT=results\rl\multigate_reliability_first\bc_v3_baseline_20
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.reliability_validation evaluate-model ^
  --kind bc ^
  --model results\rl\multigate_longrun\bc_v3_balanced_v2_20260728\bc_v3.pt ^
  --out "%OUT%" --stage C0 --mode full --cases 20 --adapter holoocean
exit /b %ERRORLEVEL%
