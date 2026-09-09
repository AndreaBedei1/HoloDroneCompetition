@echo off
setlocal EnableDelayedExpansion
if "%~1"=="" (
  echo Usage: %~nx0 RUN_DIR [ALIAS]
  exit /b 2
)
cd /d "%~dp0\.."
set RUN_DIR=%~1
set ALIAS=%~2
if "!ALIAS!"=="" set ALIAS=best_reliable
set PPO_MODEL=!RUN_DIR!\best_models\!ALIAS!.zip
if not exist "!PPO_MODEL!" (
  echo Missing PPO alias: !PPO_MODEL!
  exit /b 2
)
set BC_V3=results\rl\multigate_longrun\bc_v3_balanced_v2_20260728\bc_v3.pt
set BC_V1=results\rl_public\stage1\bc\model\best_model.pt
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%I
set BASE=!RUN_DIR!\evaluations\paired_reliability_!STAMP!
set TRACK=marine_race_arena\tracks\tests\two_gate_straight.json
set SEEDS=25400-25404

conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval ^
  --track "!TRACK!" --controller rl_multigate_controller --model "!BC_V3!" ^
  --seeds !SEEDS! --out "!BASE!\bc_v3" --adapter holoocean --current-profile none
if not exist "!BASE!\bc_v3\eval_summary.json" exit /b 1

conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval ^
  --track "!TRACK!" --controller rl_multigate_controller --model "!PPO_MODEL!" ^
  --seeds !SEEDS! --out "!BASE!\ppo_!ALIAS!" --adapter holoocean --current-profile none
if not exist "!BASE!\ppo_!ALIAS!\eval_summary.json" exit /b 1

conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval ^
  --track "!TRACK!" --controller rule_gate_center_then_commit ^
  --seeds !SEEDS! --out "!BASE!\rule" --adapter holoocean --current-profile none
if not exist "!BASE!\rule\eval_summary.json" exit /b 1

conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval ^
  --track "!TRACK!" --controller hybrid_gate_controller --model "!BC_V1!" ^
  --seeds !SEEDS! --out "!BASE!\hybrid" --adapter holoocean --current-profile none
if not exist "!BASE!\hybrid\eval_summary.json" exit /b 1

echo Paired BC-v3/PPO/rule/hybrid evidence written to !BASE!
exit /b 0
