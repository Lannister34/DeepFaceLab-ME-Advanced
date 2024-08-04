@echo off

chcp 936 > nul

title --- [Composite MOV Lossless Video]

set "filename=%~nx0"
call _internal\setenv.bat


"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed video-from-sequence ^
    --input-dir "%WORKSPACE%\data_dst\merged" ^
    --output-file "%WORKSPACE%\result.mov" ^
    --reference-file "%WORKSPACE%\data_dst.*" ^
    --include-audio ^
    --lossless

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" videoed video-from-sequence ^
    --input-dir "%WORKSPACE%\data_dst\merged_mask" ^
    --output-file "%WORKSPACE%\result_mask.mov" ^
    --reference-file "%WORKSPACE%\data_dst.*" ^
    --lossless

pause
