@echo off

chcp 936 > nul

title --- [Exporting the ME Directed Broadcast Model]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" exportdfm ^
    --model-dir "%WORKSPACE%\model" ^
    --model ME

pause
