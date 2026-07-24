@echo off
REM Resume a Stage-2 randomized PPO run. Usage: resume_stage2_ppo.bat <run-dir> [--steps N] [--arm ...]
setlocal
cd /d "%~dp0\.."
if "%~1"=="" (
  echo Usage: %~nx0 ^<run-directory^> [--steps N] [--arm bcinit_controlled^|scratch_controlled^|scratch_default]
  exit /b 2
)
echo [resume] Resuming: %~1
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --resume-run "%~1" --condition randomized --steps 6000 %2 %3 %4 %5 %6
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [resume] non-zero exit %EXIT_CODE%
  pause
)
exit /b %EXIT_CODE%
