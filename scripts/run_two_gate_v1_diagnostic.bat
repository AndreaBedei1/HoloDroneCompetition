@echo off
REM Frozen BC-v1 multi-gate diagnostic on the two-gate straight track (real HoloOcean, no
REM currents). Records tracker phase / expected beacon / referee gates and the first failure.
setlocal
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.multigate_diagnostic ^
  --track marine_race_arena/tracks/tests/two_gate_straight.json ^
  --controller rl_gate_controller --model results/rl_public/stage1/bc/model/best_model.pt ^
  --seeds 1760-1762 --out results/rl/multigate_v1/two_gate_straight --current-profile none %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" ( echo [run] non-zero exit %EXIT_CODE% & pause )
exit /b %EXIT_CODE%
