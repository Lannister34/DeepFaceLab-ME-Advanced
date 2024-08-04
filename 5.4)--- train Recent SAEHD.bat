@echo off

chcp 936 > nul

title --- [Training SAEHD continues from last time]

set "filename=%~nx0"
call _internal\setenv_old.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "%WORKSPACE%\pretrain_faces" ^
    --model-dir "%WORKSPACE%\model" ^
    --model SAEHD ^
    --silent-start

:end
echo If there is a problem, please in the QQ group:747439134 Feedback
pause
