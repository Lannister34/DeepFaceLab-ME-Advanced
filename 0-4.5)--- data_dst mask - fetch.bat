@echo off

chcp 936 > nul

title --- [DST-aligned mask extraction]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg fetch ^
    --input-dir "%WORKSPACE%\data_dst\aligned"

pause
