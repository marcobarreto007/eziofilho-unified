@echo off
:: ??? Set proper working directory ???
set "BASE=C:\Users\anapa\SuperIA\EzioFilhoUnified"
set "COMPONENT_NAME=ezio_experts"
set "SCRIPT_NAME=sentiment_expert"

:: ??? Check if directories exist ???
if not exist "%BASE%\%COMPONENT_NAME%" (
  echo Creating directory "%BASE%\%COMPONENT_NAME%"
  mkdir "%BASE%\%COMPONENT_NAME%" 2>nul
)

:: ??? Create/edit the file ???
echo Opening file for editing: %BASE%\%COMPONENT_NAME%\%SCRIPT_NAME%.py
notepad "%BASE%\%COMPONENT_NAME%\%SCRIPT_NAME%.py"

:: ??? Run the script after editing ???
echo.
echo After saving the file, the script will run.
echo.
pause
python "%BASE%\%COMPONENT_NAME%\%SCRIPT_NAME%.py"
