@echo off

chcp 936 > nul

title --- [SRC-aligned data recovery]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" util ^
    --input-dir "%WORKSPACE%\data_src\aligned" ^
    --restore-faceset-metadata

pause
