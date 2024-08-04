@echo off

chcp 936 > nul

title ---【切脸 data_dst 手动】

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" extract ^
    --input-dir "%WORKSPACE%\data_dst" ^
    --output-dir "%WORKSPACE%\data_dst\aligned" ^
    --detector manual ^
    --max-faces-from-image 0 ^
    --output-debug

pause
