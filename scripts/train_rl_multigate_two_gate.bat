@echo off
REM Train observation-v3 PPO on the current-free two-gate straight curriculum.
REM Real HoloOcean only; no fallback. The workflow creates a timestamped run.
setlocal
cd /d "%~dp0\.."
set MODEL=results/rl/multigate_v3/models/bc_v3_dagger_neutral.pt
conda run -n marine_race_rl python -m marine_race_arena.learning.model_contract_v3 validate --model "%MODEL%"
if errorlevel 1 exit /b %ERRORLEVEL%
conda run -n marine_race_rl python -m marine_race_arena.learning.train_multigate_ppo ^
  --stage R1 --track-index 0 --bc-model "%MODEL%" --steps 5000 ^
  --train-seed 20110 --eval-seeds 21030-21034 ^
  --adapter holoocean --current-profile none --max-steps 600 ^
  --checkpoint-freq 500 --eval-freq 2500 %*
set EXIT_CODE=%ERRORLEVEL%
echo [multigate] Training exit=%EXIT_CODE%
echo [multigate] Inspect results\rl\multigate_v3\r1\ppo_multigate_v3\^<timestamp^>\
exit /b %EXIT_CODE%
