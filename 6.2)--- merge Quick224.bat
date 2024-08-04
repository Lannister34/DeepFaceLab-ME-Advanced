@echo off

chcp 936 > nul

title --- [Model Quick224 Synthetic Face]

set "filename=%~nx0"
call _internal\setenv_old.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" merge ^
    --input-dir "%WORKSPACE%\data_dst" ^
    --output-dir "%WORKSPACE%\data_dst\merged" ^
    --output-mask-dir "%WORKSPACE%\data_dst\merged_mask" ^
    --aligned-dir "%WORKSPACE%\data_dst\aligned" ^
    --model-dir "%WORKSPACE%\model" ^
    --xseg-dir "%INTERNAL%\model_generic_xseg" ^
    --model Quick224

pause
