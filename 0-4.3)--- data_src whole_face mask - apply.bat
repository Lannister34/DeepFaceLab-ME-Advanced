@echo off

chcp 936 > nul

title --- [SRC-aligned using built-in masks]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg apply ^
    --input-dir "%WORKSPACE%\data_src\aligned" ^
    --model-dir "%INTERNAL%\model_generic_xseg"

pause
