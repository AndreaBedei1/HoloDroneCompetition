@echo off
REM Corrected visual-pose detector validation on real HoloOcean (stationary + moving modes,
REM per-track public aperture size, canonical modulo-180 yaw, no fallback, currents disabled).
setlocal
cd /d "%~dp0\.."
echo [run] Corrected gate-pose validation (real HoloOcean)
conda run -n marine_race_rl python -m marine_race_arena.learning.vision_pose_capture %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" ( echo [run] non-zero exit %EXIT_CODE% & pause )
exit /b %EXIT_CODE%
