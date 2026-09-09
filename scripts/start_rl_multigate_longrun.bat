@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_rl_multigate_longrun.ps1" %*
exit /b %ERRORLEVEL%
