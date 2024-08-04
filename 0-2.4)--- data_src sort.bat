@echo off

chcp 936 > nul

title --- [SRC-aligned sorting]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" sort ^
    --input-dir "%WORKSPACE%\data_src\aligned"

pause
