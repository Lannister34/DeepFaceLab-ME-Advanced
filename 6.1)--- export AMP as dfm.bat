@echo off

chcp 936 > nul

title --- [Exporting AMP Directed Broadcast Models]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" exportdfm ^
    --model-dir "%WORKSPACE%\model" ^
    --model AMP

pause
