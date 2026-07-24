@echo off
REM Stage-1 fixed-start KL calibration (real HoloOcean). BC already solves Stage 1, so this
REM only calibrates PPO update stability (KL-safe config). Args after this pass through
REM (e.g. --config kl_safe_v2). Never kills processes globally.
setlocal
cd /d "%~dp0\.."
echo [run] Repo root: %CD%
echo [run] Stage-1 KL calibration (bcinit_controlled, kl_safe_v1, 500 steps)
conda run -n marine_race_rl python -m marine_race_arena.learning.launch_stage1_ppo --arm bcinit_controlled --condition fixed --config kl_safe_v1 --steps 500 %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [run] non-zero exit %EXIT_CODE% (3 = a KL/other safety stop; inspect run_status.json)
  pause
)
exit /b %EXIT_CODE%
