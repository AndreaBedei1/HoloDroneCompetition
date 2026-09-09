@echo off
setlocal EnableDelayedExpansion
if "%~1"=="" (
  echo Usage: %~nx0 RUN_DIR
  exit /b 2
)
cd /d "%~dp0\.."
set RUN_DIR=%~1
set RL_MODEL=!RUN_DIR!\best_models\best_overall.zip
if not exist "!RL_MODEL!" set RL_MODEL=!RUN_DIR!\best_models\latest_safe.zip
if not exist "!RL_MODEL!" (
  echo No best_overall or latest_safe checkpoint exists.
  exit /b 2
)
set BC_MODEL=results/rl_public/stage1/bc/model/best_model.pt
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%I
set BASE=!RUN_DIR!\evaluations\paired_comparison_!STAMP!
for %%C in (rule_gate_center_then_commit hybrid_gate_controller rl_multigate_controller) do (
  set MODEL_ARG=
  if "%%C"=="hybrid_gate_controller" set MODEL_ARG=--model "!BC_MODEL!"
  if "%%C"=="rl_multigate_controller" set MODEL_ARG=--model "!RL_MODEL!"
  conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval ^
    --track marine_race_arena/tracks/tests/two_gate_straight.json ^
    --controller %%C !MODEL_ARG! --seeds 25100-25119 ^
    --out "!BASE!\%%C" --adapter holoocean --current-profile none
  if errorlevel 1 exit /b !ERRORLEVEL!
)
echo Comparison written to !BASE!
exit /b 0
