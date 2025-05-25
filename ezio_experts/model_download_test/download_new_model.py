@echo off
REM Test downloading a new model
set "BASE=C:\Users\anapa\SuperIA\EzioFilhoUnified\ezio_experts\"
set "COMP=model_download_test"
set "SCRIPT=download_new_model"
mkdir "%BASE%%COMP%" 2>nul
notepad "%BASE%%COMP%\%SCRIPT%.py"
py "%BASE%%COMP%\%SCRIPT%.py"