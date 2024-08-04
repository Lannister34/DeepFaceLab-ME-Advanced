@echo off

chcp 936 > nul

title --- [Training ME Designated Preset File]

set "filename=%~nx0"

call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "%WORKSPACE%\pretrain_faces" ^
    --model-dir "%WORKSPACE%\model" ^
    --model ME ^
    --config-training-file "%WORKSPACE%\model"

pause