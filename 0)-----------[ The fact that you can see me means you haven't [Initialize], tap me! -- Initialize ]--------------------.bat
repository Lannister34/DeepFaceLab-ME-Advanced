@echo off
:: Use UTF-8 encoding without BOM
chcp 936 > nul
setlocal
echo Modify registry to show/hide files

REG ADD "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced" /v Hidden /t REG_DWORD /d 2 /f > nul

echo ok!

echo Changing Notepad's font settings

reg add "HKCU\Software\Microsoft\Notepad" /v "lfFaceName" /t REG_SZ /d "Consolas" /f > nul

echo ok!

echo Changing the font settings of a command prompt

reg add "HKCU\Console" /v "FaceName" /t REG_SZ /d "Consolas" /f > nul

echo ok!

echo Graphics settings --- hardware accelerated gpu program

reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v HwSchMode /t REG_DWORD /d 2 /f > nul

echo ok!

echo Please restart your computer after a while for it to take effect (Good Lord! This thing can really speed things up)

echo.

echo Read the current directory

set "currentDir=%CD%"
:: attrib +h "%~f0" hide oneself
echo ok!
echo Restart Explorer
taskkill /f /im explorer.exe > nul
start explorer.exe
echo ok!
:: echo Wait to make sure Explorer is started
:: timeout /t 2 /nobreak > NUL
echo Shrinking all submenus
attrib +h *.*
echo ok!
echo Required documents are being displayed
attrib -h "0)++++++++++++++++[ Recent Use ]+++++++++++++++++++++++++++++++++++++++++++++.bat"
attrib -h "0)-----------[ Graphics card settings -- RG switch ]------------------------------------------------------------------.bat"
attrib -h "0)-----------[ Expand All ]--------------------------------------------------------------------------------------------.bat"
attrib -h "1)-----------[ Video Tools ]-------------------------------------------------------------------------------------------.bat"
attrib -h "2)-----------[ Data_src Tools ]------------------------------------------------------------------------------------------.bat"
attrib -h "3)-----------[ Data_dst Tools ]------------------------------------------------------------------------------------------.bat"
attrib -h "4)-----------[ XSeg Tools ]-------------------------------------------------------------------------------------------.bat"
attrib -h "5)-----------[ Train Models ]-------------------------------------------------------------------------------------------.bat"
attrib -h "6)-----------[ Merge Tools ]-------------------------------------------------------------------------------------------.bat"
attrib -h "7)-----------[ Encode Videos ]------------------------------------------------------------------------------------------.bat"
attrib -h "8)-----------[ Extra Function ]------------------------------------------------------------------------------------------.bat"
echo ok!
echo Open the DeepFaceLab catalog
start "" "%currentDir%"

echo.
echo ----------------------------------------------------------------------------------------------------------
echo 1. introductory
echo.
echo This disclaimer applies to the DeepFaceLab open source project (the "Project"). The Project is open source software designed to provide technology for face replacement and image processing. Users (hereinafter referred to as "You") should carefully read and understand all the terms of this disclaimer when using the Project.
echo.
echo 2. Scope of License
echo.
echo This project is distributed under the GPL-3.0 open source license. This license authorizes you to use, copy, modify, and distribute this project, subject to all terms and conditions of this license.
echo.
echo 3. statement denying or limiting responsibility
echo.
echo this program is provided on an "as is" and "as available" basis without warranty of any kind, either express or implied, including, but not limited to, warranties of merchantability, fitness for a particular purpose, or non-infringement. in no event shall the author or copyright owner of this program be liable for any direct, indirect, incidental, special, exemplary or consequential damages arising out of the use of this program.
echo.
echo 4. Limitations on use
echo.
echo Your use of the Program is subject to applicable laws and regulations. You undertake not to use the Program for any unlawful or unauthorized purpose, including, but not limited to, infringing the copyright, privacy or other rights of others.
echo.
echo 5. Copyright and ownership
echo.
echo The copyright and intellectual property rights of this item belong to the original author. This disclaimer does not imply transfer of any copyright or other intellectual property rights to the user.
echo.
echo 6. Final interpretation
echo.
echo The interpretation and modification of this disclaimer is the sole responsibility of the maintainer of this project. In the event of a conflict between the English and Chinese versions of this disclaimer, the English version shall prevail.
echo.
echo Chinese version: https://github.com/curios-city/DeepFaceLab
echo English version: https://github.com/MachineEditor/DeepFaceLab
echo.
echo Please select whether or not to turn on the welcome screen:

set /p choice=Enter your selection(y/n): 
set source=_internal\DeepFaceLab\utils\logo2.py
set destination=_internal\DeepFaceLab\utils\logo.py

if "%choice%"=="n" (
    echo Inside if block
    if exist %destination% (
        del %destination%
    )
    copy %source% %destination% > nul
) else (
    goto end
)

:end
endlocal
pause