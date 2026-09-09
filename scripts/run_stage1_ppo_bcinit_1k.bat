@echo off
REM Launch the Stage-1 BC-initialized PPO 1,000-step smoke (real HoloOcean).
REM Safe defaults: holoocean adapter, no fallback, fresh reset, committed public BC model.
REM Never kills HoloOcean/Unreal processes; only starts this repo's own run.
setlocal
cd /d "%~dp0\.."
echo [run] Repo root: %CD%
echo [run] Results will be written under: results\rl\stage1\ppo_bcinit\
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit --steps 1000 %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [run] FAILED with exit code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
