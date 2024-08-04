@echo off

chcp 936 > nul

title --- [Training Quick512 original model]

set "filename=%~nx0"

if not "%filename:~0,2%"=="0-" (
    copy "%~nx0" "0-%~nx0" > nul
    echo [Recently Used] is written, if you need to clear the history please delete it manually!
)

echo.
echo Warm Tip: This model requires 24G video memory to ensure full operation! If you have a 12G video card, you can try it.
echo If you have 8G RAM, you can try to turn on RG optimization (sacrificing speed to reduce RAM requirement).
echo.

call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" train ^
    --training-data-src-dir "%WORKSPACE%\data_src\aligned" ^
    --training-data-dst-dir "%WORKSPACE%\data_dst\aligned" ^
    --pretraining-data-dir "%WORKSPACE%\pretrain_faces" ^
    --pretrained-model-dir "%INTERNAL%\pretrain_Quick512" ^
    --model-dir "%WORKSPACE%\model" ^
    --model Q512

:end
echo If there is a problem, please feedback in the QQ group:747439134
pause
