@echo off
echo Iniciando EzioFilho_LLMGraph_Alpha Brain Core...

REM Configurar ambiente
set PYTHONPATH=%CD%;%PYTHONPATH%
echo Configurado PYTHONPATH: %PYTHONPATH%

REM Verificar se Python está disponível
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Python não encontrado no PATH.
    pause
    exit /b 1
)

REM Instalar dependências necessárias com feedback visual
echo Instalando dependências necessárias...
python -m pip install psutil
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao instalar psutil.
    pause
    exit /b 1
)

python -m pip install requests pandas
if %ERRORLEVEL% NEQ 0 (
    echo AVISO: Falha ao instalar requests/pandas, mas tentaremos continuar.
)

echo Dependências instaladas com sucesso!

REM Verificar se brain_core.py existe
if not exist "experts\claude_sync\brain_core.py" (
    echo ERRO: O arquivo brain_core.py não foi encontrado.
    echo Caminho esperado: %CD%\experts\claude_sync\brain_core.py
    pause
    exit /b 1
)

echo Executando o Brain Core...
python experts\claude_sync\brain_core.py --with-brain --continuous --log-level INFO

echo.
echo Execução do Brain Core concluída.
pause