@echo off
REM Paired two-gate comparison: rule vs hybrid vs learned RL, same five seeds.
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."
set RL_MODEL=results/rl/multigate_v3/r1/ppo_multigate_v3/20260727_133031/best_model/best_model.zip
set BC_MODEL=results/rl_public/stage1/bc/model/best_model.pt
conda run -n marine_race_rl python -m marine_race_arena.learning.model_contract_v3 validate --model "%RL_MODEL%"
if errorlevel 1 exit /b %ERRORLEVEL%
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%I
set BASE=results/rl/multigate_v3/manual_comparison/two_gate/!STAMP!
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval --track marine_race_arena/tracks/tests/two_gate_straight.json --controller rule_gate_center_then_commit --seeds 21200-21204 --out "!BASE!\rule" --adapter holoocean --current-profile none
set RC_RULE=!ERRORLEVEL!
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval --track marine_race_arena/tracks/tests/two_gate_straight.json --controller hybrid_gate_controller --model "%BC_MODEL%" --seeds 21200-21204 --out "!BASE!\hybrid" --adapter holoocean --current-profile none
set RC_HYBRID=!ERRORLEVEL!
conda run -n marine_race_rl python -m marine_race_arena.learning.closed_loop_eval --track marine_race_arena/tracks/tests/two_gate_straight.json --controller rl_multigate_controller --model "%RL_MODEL%" --seeds 21200-21204 --out "!BASE!\rl" --adapter holoocean --current-profile none
set RC_RL=!ERRORLEVEL!
echo ================= PAIRED SUMMARY =================
echo rule exit=!RC_RULE!
if exist "!BASE!\rule\eval_summary.json" type "!BASE!\rule\eval_summary.json"
echo hybrid exit=!RC_HYBRID!
if exist "!BASE!\hybrid\eval_summary.json" type "!BASE!\hybrid\eval_summary.json"
echo learned RL exit=!RC_RL!
if exist "!BASE!\rl\eval_summary.json" type "!BASE!\rl\eval_summary.json"
echo Output: !BASE!
echo ==================================================
exit /b 0
