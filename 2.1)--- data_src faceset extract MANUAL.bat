@echo off

chcp 936 > nul

title --- [cut face data_src manual]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" extract ^
    --input-dir "%WORKSPACE%\data_src" ^
    --output-dir "%WORKSPACE%\data_src\aligned" ^
    --detector manual

pausefix name
