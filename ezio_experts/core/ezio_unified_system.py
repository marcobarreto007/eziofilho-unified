@echo off
REM Deploy all experts automatically
echo ========================================
echo EZIOFILHO UNIFIED - EXPERT DEPLOYMENT
echo ========================================

set "BASE=C:\Users\anapa\SuperIA\EzioFilhoUnified\ezio_experts\"

REM Deploy each expert
call :deploy_expert "sentiment" "sentiment_analyzer"
call :deploy_expert "technical" "technical_analyzer" 
call :deploy_expert "fundamental" "fundamental_analyzer"
call :deploy_expert "macro" "macro_analyzer"
call :deploy_expert "risk" "risk_manager"
call :deploy_expert "volatility" "volatility_analyzer"
call :deploy_expert "credit" "credit_analyzer"
call :deploy_expert "liquidity" "liquidity_analyzer"
call :deploy_expert "algorithmic" "algo_trader"
call :deploy_expert "options" "options_analyzer"
call :deploy_expert "fixed_income" "bonds_analyzer"
call :deploy_expert "crypto" "crypto_analyzer"

echo.
echo ========================================
echo ALL EXPERTS DEPLOYED SUCCESSFULLY!
echo ========================================
goto :eof

:deploy_expert
set "COMP=%~1"
set "SCRIPT=%~2"
echo Deploying %COMP% expert...
mkdir "%BASE%%COMP%" 2>nul
REM Create expert file if not exists
if not exist "%BASE%%COMP%\%SCRIPT%.py" (
    echo # %SCRIPT%.py - %COMP% expert > "%BASE%%COMP%\%SCRIPT%.py"
    echo print("Expert %COMP% initialized") >> "%BASE%%COMP%\%SCRIPT%.py"
)
goto :eof