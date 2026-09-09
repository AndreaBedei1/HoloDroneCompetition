@echo off
REM ---------------------------------------------------------------------------
REM  Debug visivo dell'osservazione onboard -- Horseshoe Bay.
REM
REM  Apre UNA finestra con cinque pannelli e mostra solo segnali disponibili al
REM  rover. Nessun dato del simulatore entra nella modalita standard.
REM
REM    debug_horseshoe.bat                 policy appresa, dall'inizio
REM    debug_horseshoe.bat --paused        parte in pausa, avanzi con "n"
REM    debug_horseshoe.bat --driver expert guida il controller a regole
REM    debug_horseshoe.bat --video         salva anche un MP4
REM    debug_horseshoe.bat --with-ground-truth   aggiunge la striscia di
REM                                              confronto, etichettata
REM
REM  Tasti: spazio pausa | n avanti di uno | f piu veloce | s piu lento
REM         r tempo reale | c screenshot | q esci
REM ---------------------------------------------------------------------------
setlocal
set "REPO=%~dp0"
set "PY=%USERPROFILE%\.conda\envs\marine_race_rl\python.exe"

if not exist "%PY%" (
  echo [errore] interprete non trovato: %PY%
  echo          attiva l'ambiente marine_race_rl o correggi il percorso.
  exit /b 1
)

cd /d "%REPO%"
set "PYTHONPATH=%REPO%"
echo [debug] avvio Horseshoe Bay -- solo segnali onboard
"%PY%" -m marine_race_arena.tools.obs_debug --track horseshoe_bay %*
endlocal
