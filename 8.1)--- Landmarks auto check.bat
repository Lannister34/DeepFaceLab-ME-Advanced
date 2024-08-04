@echo off

chcp 936 > nul

title --- [Landmarks] --- [Author] ---[Fruit-eating beavers]

set "filename=%~nx0"
call _internal\setenv.bat
"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\ErrFaceFilter\ErrFaceFilter.py" "1"
pause
