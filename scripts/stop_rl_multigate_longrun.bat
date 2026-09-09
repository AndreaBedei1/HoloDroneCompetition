@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_rl_multigate_longrun.ps1" %*
exit /b %ERRORLEVEL%
