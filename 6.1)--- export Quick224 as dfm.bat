@echo off

chcp 936 > nul

title --- [Exporting Quick224 live broadcast model]

set "filename=%~nx0"
call _internal\setenv_old.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" exportdfm ^
    --model-dir "%WORKSPACE%\model" ^
    --model Quick224

pause
