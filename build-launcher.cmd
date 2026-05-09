@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\build-launcher.ps1" %*
if errorlevel 1 exit /b %errorlevel%
