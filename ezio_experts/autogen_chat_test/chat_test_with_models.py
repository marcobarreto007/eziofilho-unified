@echo off
REM Sentiment Analysis with RoBERTa model
set "BASE=C:\Users\anapa\SuperIA\EzioFilhoUnified\ezio_experts\"
set "COMP=sentiment_analysis_test"
set "SCRIPT=sentiment_expert_test"
mkdir "%BASE%%COMP%" 2>nul
notepad "%BASE%%COMP%\%SCRIPT%.py"
py "%BASE%%COMP%\%SCRIPT%.py"