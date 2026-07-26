@echo off
REM Official Mixed Endurance circuit, CURRENT-FREE, real HoloOcean, no fallback.
REM Mixed Endurance is a current_gate task, so a current-free run also overrides the
REM validation benchmark_task to clean_gate; geometry/gates/laps/referee are unchanged.
setlocal
cd /d "%~dp0\.."
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval ^
  --track marine_race_arena/tracks/marine_race_mixed_endurance.json ^
  --controller rule_gate_center_then_commit --seeds 1800-1804 ^
  --out results/rl_public/visual_pose_v2/official_no_current/circuit_mixed ^
  --adapter holoocean --current-profile none --benchmark-task clean_gate %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" ( echo [run] non-zero exit %EXIT_CODE% & pause )
exit /b %EXIT_CODE%
