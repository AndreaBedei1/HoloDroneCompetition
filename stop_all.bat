@echo off
REM ---------------------------------------------------------------------------
REM  Chiude tutto cio che appartiene a questa campagna RL, e nient'altro.
REM
REM  Su questa macchina girano anche altri progetti (un worker in
REM  Desktop\Instagram, ComfyUI). Il filtro e sul percorso dell'eseguibile:
REM  vengono chiusi solo i python dell'ambiente marine_race_rl e gli engine
REM  HoloOcean che ne discendono. Nessun file viene cancellato.
REM ---------------------------------------------------------------------------
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$env_marker = 'marine_race_rl';" ^
  "$py = Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%%'\" | Where-Object { $_.ExecutablePath -like ('*' + $env_marker + '*') };" ^
  "if ($py) { $py | ForEach-Object { Write-Host ('chiudo python ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } } else { Write-Host 'nessun python della campagna attivo' };" ^
  "Start-Sleep -Seconds 2;" ^
  "$eng = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'Holodeck|HoloOcean' };" ^
  "if ($eng) { $eng | ForEach-Object { Write-Host ('chiudo engine ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } } else { Write-Host 'nessun engine attivo' };" ^
  "Start-Sleep -Seconds 1;" ^
  "Write-Host '--- verifica finale ---';" ^
  "$r = Get-CimInstance Win32_Process | Where-Object { ($_.ExecutablePath -like ('*' + $env_marker + '*')) -or ($_.Name -match 'Holodeck|HoloOcean') };" ^
  "if ($r) { $r | ForEach-Object { Write-Host ('RESIDUO: ' + $_.ProcessId + ' ' + $_.Name) } } else { Write-Host 'nessun processo residuo della campagna' };" ^
  "Write-Host '';" ^
  "Write-Host 'lasciati intatti (altri progetti):';" ^
  "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%%'\" | ForEach-Object { Write-Host ('  ' + $_.ProcessId + '  ' + $(if ($_.ExecutablePath) { $_.ExecutablePath } else { '(path non leggibile)' })) }"
endlocal
