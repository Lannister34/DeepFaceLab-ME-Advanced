@echo off

chcp 936 > nul

title --- [Exporting Quick512 live broadcast model]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" exportdfm ^
    --model-dir "%WORKSPACE%\model" ^
    --model Q512

pause
