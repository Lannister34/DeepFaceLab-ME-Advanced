@echo off

chcp 936 > nul

title --- [Training ME Designated Preset File]

set "filename=%~nx0"
echo Usage Note: The configuration generated and read here is taken from the full yaml file path!

echo If the parameter is a folder, then \def_conf_file.yaml is automatically generated or read.

call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "%WORKSPACE%\pretrain_faces" ^
    --model-dir "%WORKSPACE%\model" ^
    --model ME ^
    --config-training-file "%WORKSPACE%\model"

pause
