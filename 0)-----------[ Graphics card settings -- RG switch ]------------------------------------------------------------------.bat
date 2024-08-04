@echo off
setlocal
chcp 936 > nul
rem Check if the configuration file exists
cd /d "%~dp0"
if exist _internal\config.txt (
    rem If a configuration file exists, read the selection from the configuration file
    < _internal\config.txt (
        set /p var1=
        set /p var2=
    )

    call :tell
    call :choose
    call :tell
    call :do

) else (
    echo 首次写入配置

    call :choose
    call :tell
    call :do
)
goto end

:choose

echo.

echo Please select the graphics card option:

echo 1. DML (generic, supports AMD graphics cards)

echo 2. CUDA (NVIDIA)

echo.

set /p var1=Enter your selection (1 or 2): 

echo.

echo Whether to enable RG optimization (slower training, lower video memory requirements):

echo 1. Open RG Optimization

echo 2. Turn off RG optimization

echo.

set /p var2=Enter your choice (1 or 2).

echo.
echo.
goto :eof


:tell


if %var1% == 1 (
    cd "_internal\python_common\Lib\site-packages\"
    call dml.bat
    echo ------------------------------------------------You have selected DML
) else if %var1% == 2 (
    cd "_internal\python_common\Lib\site-packages\"
    call cuda.bat
    echo ------------------------------------------------You have selected CUDA
) else (
    echo ------------------------------------------------Graphics cards: an ineffective choice
    goto end
)

cd /d "%~dp0"

if %var2% == 1 (
    set source2=_internal\DeepFaceLab\core\leras\archis\DeepFakeArchi_rg.py
    echo ------------------------------------------------RG optimization has been turned on
) else if %var2% == 2 (
    set source2=_internal\DeepFaceLab\core\leras\archis\DeepFakeArchi_old.py
    echo ------------------------------------------------Closed RG optimization
) else (
    echo ------------------------------------------------RG: Invalid choices
    goto end
)

goto :eof

:do

cd /d "%~dp0"

set destination2=_internal\DeepFaceLab\core\leras\archis\DeepFakeArchi.py

if exist %destination2% (
    del %destination2%
)

copy %source2% %destination2% > nul
echo RG file replacement is complete!

echo.

rem Saving user selections to a configuration file
cd /d "%~dp0"
echo %var1: =% >_internal\config.txt

echo %var2: =% >>_internal\config.txt
goto :eof

:end
endlocal

pause
