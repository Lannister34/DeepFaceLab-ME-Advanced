@echo off

chcp 936 > nul

title --- [Training ME models to open the WEB panel]

set "filename=%~nx0"
echo Please note that the original dfl model can be upgraded to the mve version for training, but it is difficult to go back. Be careful with your backups!

echo or choose 5.4) --- train SAEHDLegacy

echo.

echo Note: The yaml files generated and read here are named after the models in the model folder!

echo For example: workspace\model\new_ME_configuration_file.yaml

call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "%WORKSPACE%\pretrain_faces" ^
    --model-dir "%WORKSPACE%\model" ^
    --model ME ^
    --flask-preview ^
    --auto-gen-config

pause
