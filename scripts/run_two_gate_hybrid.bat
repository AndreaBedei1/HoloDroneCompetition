@echo off
REM Hybrid (rule backbone + BC-v1 visual servo) multi-gate diagnostic on the two-gate straight
REM track (real HoloOcean, no currents).
setlocal
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.multigate_diagnostic ^
  --track marine_race_arena/tracks/tests/two_gate_straight.json ^
  --controller hybrid_gate_controller --model results/rl_public/stage1/bc/model/best_model.pt ^
  --seeds 1768-1769 --out results/rl/multigate_v1/hybrid_two_gate --current-profile none %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" ( echo [run] non-zero exit %EXIT_CODE% & pause )
exit /b %EXIT_CODE%
