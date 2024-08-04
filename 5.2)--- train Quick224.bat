@echo off

chcp 936 > nul

title --- [Training Quick224 original model]

set "filename=%~nx0"
call _internal\setenv_old.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "%WORKSPACE%\pretrain_faces" ^
    --pretrained-model-dir "%INTERNAL%\pretrain_Quick224" ^
    --model-dir "%WORKSPACE%\model" ^
    --model Quick224

:end
echo If there is a problem, please feedback in the QQ group:747439134
pause
