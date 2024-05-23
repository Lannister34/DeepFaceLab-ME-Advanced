@echo off

chcp 65001 > nul

title ---【DST-aligned 应用自定遮罩】---【神农汉化】---【QQ交流群 747439134】

set "filename=%~nx0"

if not "%filename:~0,2%"=="0-" (
    copy "%~nx0" "0-%~nx0" > nul
    echo [最近使用] 已写入，如果需要清空历史请手动删除！
)

echo.

call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py" xseg apply ^
    --input-dir "%WORKSPACE%\data_dst\aligned" ^
    --model-dir "%WORKSPACE%\xseg_model"

pause