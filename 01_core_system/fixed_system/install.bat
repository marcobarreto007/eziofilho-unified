@echo off
echo ===============================================
echo   SISTEMA AUTOGEN 1000 - INSTALACAO COMPLETA
echo ===============================================
echo.

REM Verifica Python
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado!
    echo Instale Python 3.8+ primeiro.
    pause
    exit /b 1
)

echo [OK] Python encontrado
echo.

REM Cria ambiente virtual
echo [1/5] Criando ambiente virtual...
if exist venv_autogen (
    echo Ambiente virtual ja existe. Removendo...
    rmdir /s /q venv_autogen
)
py -m venv venv_autogen

REM Ativa ambiente virtual
echo [2/5] Ativando ambiente virtual...
call venv_autogen\Scripts\activate.bat

REM Atualiza pip
echo [3/5] Atualizando pip...
python -m pip install --upgrade pip wheel setuptools

REM Instala dependencias
echo [4/5] Instalando dependencias...
echo.

REM AutoGen e dependencias base
pip install pyautogen==0.2.18
pip install openai>=1.3
pip install python-dotenv
pip install colorama
pip install requests

REM Para modelos locais
echo Instalando suporte para modelos locais...
pip install llama-cpp-python==0.2.90
pip install transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu

REM Ollama client
pip install ollama

REM Ferramentas extras
pip install rich
pip install httpx

echo.
echo [5/5] Criando estrutura de pastas...

REM Cria pastas necessarias
if not exist "cache" mkdir cache
if not exist "logs" mkdir logs
if not exist "config" mkdir config

REM Cria arquivo de configuracao
echo Criando arquivo de configuracao...
(
echo # Configuracao do Sistema AutoGen 1000
echo.
echo # Servidor LM Studio
echo LM_STUDIO_URL=http://localhost:1234/v1
echo.
echo # Servidor Ollama
echo OLLAMA_URL=http://localhost:11434
echo.
echo # OpenAI API (opcional)
echo OPENAI_API_KEY=
echo.
echo # Modelo padrao
echo DEFAULT_MODEL=mistral
echo.
echo # Configuracoes de log
echo LOG_LEVEL=INFO
) > config\.env

echo.
echo ===============================================
echo   INSTALACAO CONCLUIDA COM SUCESSO!
echo ===============================================
echo.
echo Proximos passos:
echo.
echo 1. Instale um servidor de modelos:
echo    - LM Studio: https://lmstudio.ai/
echo    - Ollama: https://ollama.ai/
echo.
echo 2. Baixe modelos GGUF para:
echo    %USERPROFILE%\.cache\models\
echo.
echo 3. Execute o sistema:
echo    venv_autogen\Scripts\activate
echo    python main.py --test
echo.
pause