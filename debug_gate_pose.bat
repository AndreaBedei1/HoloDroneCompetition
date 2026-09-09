@echo off
REM Visual validation only. This script never starts BC or PPO.
setlocal
cd /d "%~dp0"
set "PY=%USERPROFILE%\.conda\envs\marine_race_rl\python.exe"

if not exist "%PY%" (
  echo [errore] interprete non trovato: %PY%
  echo          attiva l'ambiente marine_race_rl o correggi il percorso.
  exit /b 1
)

echo [gate-pose] Horseshoe Bay - controller e stima SOLO ONBOARD
echo [gate-pose] SPACE pausa, N/RIGHT avanti, S screenshot, Q/ESC esci
"%PY%" -m marine_race_arena.tools.gate_pose_debug --track horseshoe_bay %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
