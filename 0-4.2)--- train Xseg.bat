@echo off

chcp 936 > nul

title --- [SRC+DST Training Masking Model]

set "filename=%~nx0"
call _internal\setenv_old.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "%WORKSPACE%\pretrain_faces" ^
    --model-dir "%WORKSPACE%\xseg_model" ^
    --model XSeg

pause
