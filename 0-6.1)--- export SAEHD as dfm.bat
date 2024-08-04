@echo off

chcp 936 > nul

title --- [Export SAEHD live broadcast model]

set "filename=%~nx0"
call _internal\setenv_old.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" exportdfm ^
    --model-dir "%WORKSPACE%\model" ^
    --model SAEHD

pause
