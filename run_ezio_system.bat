@echo off
cls
color 0A
set PYTHON=C:\Users\anapa\AppData\Local\Programs\Python\Python311\python.exe

echo ========================================
echo    🤖 EZIOFILHO AI SYSTEM v2.0 🤖
echo ========================================
echo.
echo Python: %PYTHON%
echo.
echo Escolha uma opcao:
echo 1. Instalar Dependencias
echo 2. Chat Rapido (ezio_fast_chat)
echo 3. Sistema Completo (ezio_complete_system)
echo 4. Verificador de Fatos (duckduckgo)
echo 5. Sistema com Patches (ezio_system_patched)
echo 6. Sair
echo.
set /p choice=Digite sua escolha (1-6): 

if "%choice%"=="1" (
    echo.
    echo 📦 Instalando dependencias...
    "%PYTHON%" -m pip install transformers torch langchain openai colorama requests duckduckgo-search
    pause
    run_ezio_system.bat
)

if "%choice%"=="2" "%PYTHON%" ezio_fast_chat.py
if "%choice%"=="3" "%PYTHON%" ezio_complete_system_fixed.py
if "%choice%"=="4" "%PYTHON%" duckduckgo_fact_checker.py
if "%choice%"=="5" "%PYTHON%" ezio_system_patched.py
if "%choice%"=="6" exit

pause
run_ezio_system.bat