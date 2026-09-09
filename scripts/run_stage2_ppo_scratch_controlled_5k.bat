@echo off
REM Stage-2 randomized 5,000-step diagnostic: scratch_controlled (real HoloOcean, fresh reset, no fallback).
REM Development seeds 1410-1419 for checkpoint selection. Never kills processes globally.
setlocal
cd /d "%~dp0\.."
echo [run] Repo root: %CD%
echo [run] Stage-2 randomized 5k: scratch_controlled
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm scratch_controlled --condition randomized --steps 5000 %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [run] non-zero exit %EXIT_CODE% (3 = a KL/other safety stop; inspect run_status.json)
  pause
)
exit /b %EXIT_CODE%
