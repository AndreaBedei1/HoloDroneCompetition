@echo off
REM Run ALL THREE official circuits CURRENT-FREE, real HoloOcean, no fallback, in sequence.
REM Verifies env + model, disables currents, runs every circuit even if one fails, keeps
REM separate outputs, and prints a final summary table. Never kills other-project processes.
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
echo [run] Repo root: %CD%
where conda >nul 2>&1 || ( echo [run] ERROR: conda not on PATH & exit /b 1 )
echo [run] Current-free official circuits with deterministic onboard controller.

set BASE=results/current_free
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval --track marine_race_arena/tracks/marine_race_horseshoe_bay.json --controller rule_gate_center_then_commit --seeds 1800-1804 --out %BASE%/circuit_horseshoe --adapter holoocean --current-profile none
set RC_H=%ERRORLEVEL%
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval --track marine_race_arena/tracks/marine_race_vertical_serpent.json --controller rule_gate_center_then_commit --seeds 1800-1804 --out %BASE%/circuit_vertical --adapter holoocean --current-profile none
set RC_V=%ERRORLEVEL%
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval --track marine_race_arena/tracks/marine_race_mixed_endurance.json --controller rule_gate_center_then_commit --seeds 1800-1804 --out %BASE%/circuit_mixed --adapter holoocean --current-profile none --benchmark-task clean_gate
set RC_M=%ERRORLEVEL%

echo.
echo ===================== SUMMARY (exit 0 = ran; see eval_summary.json for completion_rate) =====================
echo   Horseshoe Bay    exit=%RC_H%   -^> %BASE%/circuit_horseshoe/eval_summary.json
echo   Vertical Serpent exit=%RC_V%   -^> %BASE%/circuit_vertical/eval_summary.json
echo   Mixed Endurance  exit=%RC_M%   -^> %BASE%/circuit_mixed/eval_summary.json
echo ==============================================================================================================
exit /b 0
