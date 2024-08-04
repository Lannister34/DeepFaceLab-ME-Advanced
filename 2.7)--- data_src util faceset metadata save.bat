@echo off

chcp 936 > nul

title --- [SRC-aligned Data Saving]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" util ^
    --input-dir "%WORKSPACE%\data_src\aligned" ^
    --save-faceset-metadata

pause
