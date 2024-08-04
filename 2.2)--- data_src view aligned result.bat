@echo off

chcp 936 > nul

title ---[SRC-aligned Preview]

set "filename=%~nx0"

if not "%filename:~0,2%"=="0-" (
    copy "%~nx0" "0-%~nx0" > nul
    echo [Recently Used] is written, if you need to clear the history please delete it manually!
)

echo The image viewer starts slowly, please wait 1 minute.

call _internal\setenv.bat

start "" /D "%XNVIEWMP_PATH%" /LOW "%XNVIEWMP_PATH%\xnviewmp.exe" "%WORKSPACE%\data_src\aligned"
