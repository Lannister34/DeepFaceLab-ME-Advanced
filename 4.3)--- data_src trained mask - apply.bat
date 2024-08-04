@echo off

chcp 936 > nul

title --- [SRC-aligned Application Custom Mask]

set "filename=%~nx0"
call _internal\setenv.bat
echo Please put the mask model into workspace\xseg_model\

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg apply ^
    --input-dir "%WORKSPACE%\data_src\aligned" ^
    --model-dir "%WORKSPACE%\xseg_model"

pause
