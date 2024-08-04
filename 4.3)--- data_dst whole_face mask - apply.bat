@echo off

chcp 936 > nul

title --- [DST-aligned Using the built-in mask]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg apply ^
    --input-dir "%WORKSPACE%\data_dst\aligned" ^
    --model-dir "%INTERNAL%\model_generic_xseg"

pause
