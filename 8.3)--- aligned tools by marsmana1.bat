@echo off

chcp 936 > nul

title --- [Aligned Angle Distribution] --- [Author] ---[marsmana1]

set "filename=%~nx0"
call _internal\setenv.bat

cd _internal\facesets\
"%PYTHON_EXECUTABLE%" facesets.py

pause
