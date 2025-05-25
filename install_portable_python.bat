@echo off
cls
color 0A
echo ================================================================================
echo                    PORTABLE PYTHON INSTALLER FOR EZIOFILHO
echo ================================================================================
echo.

set "PORTABLE_DIR=C:\Users\anapa\eziofilho-unified\portable_python"
set "DOWNLOAD_URL=https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip"
set "PIP_URL=https://bootstrap.pypa.io/get-pip.py"

echo [1] Creating portable Python directory...
mkdir "%PORTABLE_DIR%" 2>nul
cd "%PORTABLE_DIR%"

echo.
echo [2] Downloading Python embedded package...
echo This may take a few minutes...
powershell -Command "Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile 'python-embedded.zip'"

echo.
echo [3] Extracting Python...
powershell -Command "Expand-Archive -Path 'python-embedded.zip' -DestinationPath '.' -Force"

echo.
echo [4] Setting up Python configuration...
echo import site >> python311._pth

echo.
echo [5] Downloading pip installer...
powershell -Command "Invoke-WebRequest -Uri '%PIP_URL%' -OutFile 'get-pip.py'"

echo.
echo [6] Installing pip...
"%PORTABLE_DIR%\python.exe" get-pip.py

echo.
echo [7] Creating launcher script...
(
echo @echo off
echo set "PYTHONHOME=%PORTABLE_DIR%"
echo set "PYTHONPATH=%PORTABLE_DIR%;%PORTABLE_DIR%\Scripts"
echo set "PATH=%PORTABLE_DIR%;%PORTABLE_DIR%\Scripts;%%PATH%%"
echo.
echo echo Portable Python Ready!
echo echo Python: %PORTABLE_DIR%\python.exe
echo echo Pip: %PORTABLE_DIR%\Scripts\pip.exe
echo echo.
echo.
echo if "%%1"=="" (
echo     cmd /k
echo ) else (
echo     "%PORTABLE_DIR%\python.exe" %%*
echo )
) > "%BASE%run_python.bat"

echo.
echo ================================================================================
echo                              INSTALLATION COMPLETE!
echo ================================================================================
echo.
echo Portable Python installed at: %PORTABLE_DIR%
echo.
echo To use Python, run: run_python.bat
echo To install packages: run_python.bat -m pip install package_name
echo.
echo Testing installation...
"%PORTABLE_DIR%\python.exe" --version
echo.
pause