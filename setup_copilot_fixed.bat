@echo off
REM 1) Adiciona GitHub CLI ao PATH nesta sessão (imediato) e persiste para futuras
set "PATH=C:\Program Files\GitHub CLI;%PATH%"
setx PATH "%PATH%"

REM 2) Verifica se 'gh' está disponível
gh --version

REM 3) Instala a extensão Copilot CLI com o ID correto (deve começar com 'gh-')
gh extension install github/gh-copilot

REM 4) Cria atalhos para os comandos Copilot
doskey gcs=gh copilot suggest $*
doskey gce=gh copilot explain $*

REM 5) Testa o subcomando Copilot
gh copilot --version
