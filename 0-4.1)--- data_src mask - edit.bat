@echo off

chcp 936 > nul

title --- [SRC-aligned picture shielding]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg editor ^
    --input-dir "%WORKSPACE%\data_src\aligned"

if %errorlevel% NEQ 0 (
  pause
)
