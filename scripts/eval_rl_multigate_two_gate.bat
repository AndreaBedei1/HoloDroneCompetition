@echo off
REM Reproduce the locked 10-seed current-free two-gate evaluation.
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
set MODEL=results/rl/multigate_v3/r1/ppo_multigate_v3/20260727_133031/best_model/best_model.zip
conda run -n marine_race_rl python -m marine_race_arena.learning.model_contract_v3 validate --model "%MODEL%"
if errorlevel 1 exit /b %ERRORLEVEL%
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%I
set OUT=results/rl/multigate_v3/manual_eval/two_gate/!STAMP!
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval ^
  --track marine_race_arena/tracks/tests/two_gate_straight.json ^
  --controller rl_multigate_controller --model "%MODEL%" ^
  --seeds 21200-21209 --out "!OUT!" ^
  --adapter holoocean --current-profile none %*
set EXIT_CODE=%ERRORLEVEL%
if exist "!OUT!\eval_summary.json" (
  echo [multigate] Summary:
  type "!OUT!\eval_summary.json"
)
echo [multigate] Output: !OUT!
exit /b %EXIT_CODE%
