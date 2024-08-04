@echo off
chcp 936 > nul
title --- [Conversion of SAEHD model format]

set "filename=%~nx0"
echo Please select the serial number of the model to be converted from SAEHD to ME format! Note that it is a one-way conversion, please make your own backup!
echo If you want to convert from ME to SAEHD, it is recommended that you first reduce the learning rate to 2e-05 in ME, turn on super distortion, and iterate 5000 times. Then change the file name to SAEHD by yourself!

setlocal enabledelayedexpansion

set "workspace=workspace\model"
set "pattern=*_SAEHD_data.dat"

REM traverses the directory, finds the files that match the criteria and saves them to the array

REM sets a variable count and initializes it to 0
set /a count=0

REM traverses the files in the specified directory that match the specified pattern.
for %%F in ("%workspace%\*_SAEHD_data.dat") do (

    REM gets the filename part and replaces the "_SAEHD_data" string with null.
    set "file=%%~nF"
    set "file=!file:_SAEHD_data=!"

    REM Output file number and processed file name
    echo 	[!count!]	:  !file!

    REM stores the processed filenames in an array
    set "files[!count!]=!file!"

    REM 递增计数器
    set /a count+=1
)

REM User choice serial number
set /p choice="Enter serial number: "

REM Validates user input
if not defined files[%choice%] (
    echo Invalid selection. Exit the script.
    exit /b 1
)


REM iterates over files in a specified directory that match a specific naming pattern
for %%F in ("%workspace%\!files[%choice%]!_SAEHD_*.*") do (

    REM 获取原始文件名部分
    set "newName=%%~nxF"

    REM Replace "_SAEHD_" with "_ME_" in the file name.
    set "newName=!newName:_SAEHD_=_ME_!"

    ren "%%F" "!newName!"

    REM Outputs renaming information
    echo %%~nxF 到 !newName!
)

REM Disable Delay Extension
endlocal

echo Please take care of the backup! At this point, change the filename from ME to SAEHD to return to the original format.

echo Once you have trained with ME and saved the model, you will lose a lot of iteration time by going back to the original SAEHD!

pause
