@echo off

chcp 936 > nul

title --- [DST-aligned check error]

set "filename=%~nx0"
echo 本程序是向data_dst\aligned文件夹添加landmarks-debug图片文件供手动检查

echo.

echo If you need to check automatically, use 8.1) --- Landmarks auto check

call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" util ^
    --input-dir "%WORKSPACE%\data_dst\aligned" ^
    --add-landmarks-debug-images

pause
