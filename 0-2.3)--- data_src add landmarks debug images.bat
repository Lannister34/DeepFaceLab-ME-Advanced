@echo off

chcp 936 > nul

title --- [SRC-aligned check error]

set "filename=%~nx0"
echo This program adds landmarks-debug image files to the data_src\aligned folder for manual checking.

echo.

echo If automatic checking is required, use the 8.1)--- Landmarks自动识错 ------------------------ Landmarks auto check

call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" util ^
    --input-dir "%WORKSPACE%\data_src\aligned" ^
    --add-landmarks-debug-images

pause
