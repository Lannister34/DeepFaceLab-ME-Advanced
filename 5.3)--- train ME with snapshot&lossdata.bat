@echo off

chcp 936 > nul

title --- [ME Model Export Loss]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "%WORKSPACE%\pretrain_faces" ^
    --model-dir "%WORKSPACE%\model" ^
    --model ME --reduce-clutter --gen-snapshot
echo Files have been exported to \workspace\model\<model name>_SAEHD_state_history\date\dst_state.json and src_state.json
pause
