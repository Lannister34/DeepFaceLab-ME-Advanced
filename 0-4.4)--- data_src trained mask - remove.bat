@echo off

chcp 936 > nul

title --- [SRC-aligned Remove Mask]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg remove ^
    --input-dir "%WORKSPACE%\data_src\aligned"

pause
