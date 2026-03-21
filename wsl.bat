@echo off
echo "Caught by wsl.bat"
C:\Windows\System32\wsl.exe --exec bash %*
