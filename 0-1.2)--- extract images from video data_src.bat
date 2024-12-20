@echo off

chcp 936 > nul

title --- [Extract SRC video picture frame]

set "filename=%~nx0"
call _internal\setenv.bat

mkdir "%WORKSPACE%\data_src" 2>nul

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed extract-video ^
    --input-file "%WORKSPACE%\data_src.*" ^
    --output-dir "%WORKSPACE%\data_src"

pause
