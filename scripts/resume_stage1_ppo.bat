@echo off
REM Resume an existing Stage-1 PPO run to a larger total step budget (real HoloOcean).
REM Usage: resume_stage1_ppo.bat <run-directory> [--steps N] [extra launcher args]
REM Example: scripts\resume_stage1_ppo.bat results\rl\stage1\ppo_bcinit\20260724_101500 --steps 1500
REM Never kills HoloOcean/Unreal processes; only resumes this repo's own run.
setlocal
cd /d "%~dp0\.."
if "%~1"=="" (
  echo Usage: %~nx0 ^<run-directory^> [--steps N]
  echo Example: scripts\resume_stage1_ppo.bat results\rl\stage1\ppo_bcinit\^<timestamp^> --steps 1500
  exit /b 2
)
echo [resume] Repo root: %CD%
echo [resume] Resuming run: %~1
REM Default to 1500 total steps; any --steps passed after the run dir overrides it (argparse: last wins).
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --resume-run "%~1" --steps 1500 %2 %3 %4 %5 %6
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [resume] FAILED with exit code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
