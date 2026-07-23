@echo off
REM Launch the Stage-1 PPO-from-scratch 1,000-step control smoke (real HoloOcean).
REM Same track/seeds/config as the BC-init arm, minus the imitation warm-start.
REM Never kills HoloOcean/Unreal processes; only starts this repo's own run.
setlocal
cd /d "%~dp0\.."
echo [run] Repo root: %CD%
echo [run] Results will be written under: results\rl\stage1\ppo_scratch\
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm scratch --steps 1000 %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [run] FAILED with exit code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
