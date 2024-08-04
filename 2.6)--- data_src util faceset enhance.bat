@echo off

chcp 936 > nul

title --- [SRC-aligned high-definition restoration]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" facesettool enhance  ^
    --input-dir "%WORKSPACE%\data_src\aligned"

pause
