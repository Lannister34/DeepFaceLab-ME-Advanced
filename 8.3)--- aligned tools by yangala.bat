@echo off

chcp 936 > nul

title --- [Aligned Angle Collation] --- [Author] --- [yangala]

set "filename=%~nx0"
echo Three ways to open this tool:
echo.

echo 1: Drag the aligned folder to the bat file icon

echo 2: Drag the aligned folder into the cmd window

echo 3: Copy the path to the signed folder and paste it into the cmd window
echo.
echo.

cd /d %~dp0
call _internal\setenv.bat


set var=%~dp1

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\yaw_image_filter.py" "%~1

pause
