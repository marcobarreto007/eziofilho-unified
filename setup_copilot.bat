@echo off
REM ======================================================
REM PASSO 1: INTEGRAÇÃO GITHUB COPILOT CLI
REM ======================================================

echo ========================================
echo  CONFIGURANDO GITHUB COPILOT CLI
echo ========================================
echo.

REM 1) Verificar se GitHub CLI está instalado
echo [1/5] Verificando GitHub CLI...
gh --version
if %errorlevel% neq 0 (
    echo ERRO: GitHub CLI não encontrado!
    echo Baixe em: https://cli.github.com/
    pause
    exit /b 1
)

echo [2/5] GitHub CLI encontrado!
echo.

REM 2) Autenticar com GitHub
echo [3/5] Iniciando autenticacao GitHub...
gh auth login

REM 3) Instalar extensão Copilot CLI
echo [4/5] Instalando GitHub Copilot CLI...
gh extension install github/gh-copilot

REM 4) Criar aliases úteis
echo [5/5] Criando aliases...
doskey copilot=gh copilot $*
doskey gcs=gh copilot suggest $*
doskey gce=gh copilot explain $*

echo.
echo ========================================
echo  TESTANDO INTEGRAÇÃO
echo ========================================

REM 5) Teste básico (sem gerar código)
echo Testando comando básico...
gh copilot --help

echo.
echo ========================================
echo  INTEGRAÇÃO COMPLETA!
echo ========================================
echo Aliases criados:
echo  - copilot   (comando completo)
echo  - gcs       (suggest - gerar código)  
echo  - gce       (explain - explicar código)
echo ========================================
pause