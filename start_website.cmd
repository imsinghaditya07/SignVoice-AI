@echo off
echo Starting SignVoiceAI Modern Web Interface...

:: Start the Python AI Backend explicitly on Port 5002 in a new window
echo Launching AI Translation Backend...
start "SignVoiceAI API Backend" cmd /k "cd website && python api.py"

:: Give the server a 4 second head start to load TensorFlow safely
timeout /t 4 /nobreak >nul

:: Start the local web frontend server on Port 8000
echo Launching Frontend Interface...
start "SignVoiceAI UI Frontend" cmd /k "cd website && python -m http.server 8000"

:: Automatically launch your default browser exactly to the site!
echo Opening browser...
start http://localhost:8000

echo Both servers are fully launched! 
echo NOTE: Please keep the two new popup terminal windows open while you are testing the app!
pause

