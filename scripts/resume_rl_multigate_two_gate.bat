@echo off
REM Resume an existing two-gate PPO run to a new TOTAL step target.
setlocal
cd /d "%~dp0\.."
if "%~1"=="" (
  echo Usage: %~nx0 ^<run-dir^> [total-steps] [extra train_multigate_ppo args]
  exit /b 2
)
set RUN_DIR=%~1
set TOTAL_STEPS=%~2
if "%TOTAL_STEPS%"=="" set TOTAL_STEPS=10000
set MODEL=results/rl/multigate_v3/models/bc_v3_dagger_neutral.pt
conda run -n marine_race_rl python -m marine_race_arena.learning.model_contract_v3 validate --model "%MODEL%"
if errorlevel 1 exit /b %ERRORLEVEL%
conda run -n marine_race_rl python -m marine_race_arena.learning.train_multigate_ppo ^
  --stage R1 --track-index 0 --bc-model "%MODEL%" --resume ^
  --run-dir "%RUN_DIR%" --steps %TOTAL_STEPS% ^
  --train-seed 20110 --eval-seeds 21030-21034 ^
  --adapter holoocean --current-profile none --max-steps 600 ^
  --checkpoint-freq 500 --eval-freq 2500 %3 %4 %5 %6 %7 %8 %9
set EXIT_CODE=%ERRORLEVEL%
echo [multigate] Resume exit=%EXIT_CODE%
if exist "%RUN_DIR%\best_model\best_metrics.json" type "%RUN_DIR%\best_model\best_metrics.json"
exit /b %EXIT_CODE%
