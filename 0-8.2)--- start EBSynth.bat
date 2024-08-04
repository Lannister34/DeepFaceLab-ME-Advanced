@echo off

chcp 936 > nul

title --- [Start EBS]

set "filename=%~nx0"
call _internal\setenv.bat

start "" /D "%INTERNAL%\EbSynth" /LOW "%INTERNAL%\EbSynth\EbSynth.exe" "%INTERNAL%\EbSynth\SampleProject\sample.ebs"
