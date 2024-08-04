@echo off

chcp 936 > nul

title --- [Face cutting data_dst semi-automatic]

set "filename=%~nx0"
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" extract ^
    --input-dir "%WORKSPACE%\data_dst" ^
    --output-dir "%WORKSPACE%\data_dst\aligned" ^
    --output-debug ^
    --detector s3fd ^
    --max-faces-from-image 0 ^
    --manual-fix

pause
