@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0status_rl_multigate_longrun.ps1" %*
exit /b %ERRORLEVEL%
