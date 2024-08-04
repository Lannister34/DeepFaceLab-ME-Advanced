@echo off

chcp 936 > nul

title --- [DST-aligned scaling size]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" facesettool resize ^
    --input-dir "%WORKSPACE%\data_dst\aligned"

pause
