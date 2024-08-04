@echo off

chcp 936 > nul

title --- [Noise Reduction data_dst Picture Frame]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed denoise-image-sequence ^
    --input-dir "%WORKSPACE%\data_dst"

pause
