@echo off
title Starting Servers and Main Script
echo Starting HTTP and HTTPS servers...

:: Save current directory
set ROOTDIR=%cd%

:: Start HTTP server from modules directory
pushd modules
start "" cmd /k "python serve_http.py"
popd

:: Start HTTPS server from modules directory
pushd modules
start "" cmd /k "python serve_https.py"
popd

echo Waiting 5 seconds for servers to initialize...
timeout /t 5 /nobreak

echo 🌐  Opening browser to HTTPS server…
start "" "https://localhost:5000"

echo 🚀  Launching main.py…
start "" cmd /k "python main.py"

echo ✅  Alfred is running...
pause
