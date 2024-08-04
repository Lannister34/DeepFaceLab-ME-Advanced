@echo off

chcp 936 > nul

title --- [DST-aligned file name recovery]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" util ^
    --input-dir "%WORKSPACE%\data_dst\aligned" ^
    --recover-original-aligned-filename

pause
