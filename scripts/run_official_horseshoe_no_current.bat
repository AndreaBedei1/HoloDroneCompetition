@echo off
REM Official Horseshoe Bay circuit, CURRENT-FREE, real HoloOcean, no fallback.
REM Best onboard controller (deterministic rule backbone). Referee/geometry unchanged.
setlocal
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval ^
  --track marine_race_arena/tracks/marine_race_horseshoe_bay.json ^
  --controller rule_gate_center_then_commit --seeds 1800-1804 ^
  --out results/current_free/circuit_horseshoe ^
  --adapter holoocean --current-profile none %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" ( echo [run] non-zero exit %EXIT_CODE% & pause )
exit /b %EXIT_CODE%
