@echo off

chcp 936 > nul

title --- [DST-aligned Remove Mask]

set "filename=%~nx0"
echo Attention! Xseg information for hand drawn markers will be cleared! Please back up this part of the material!

pause

echo.

call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg remove_labels ^
    --input-dir "%WORKSPACE%\data_dst\aligned"

pause
