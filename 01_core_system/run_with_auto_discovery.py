#!/usr/bin/env python3
"""
Script para iniciar o sistema com descoberta automática de modelos ativada
"""
import os
import sys
import logging
from pathlib import Path

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s – %(message)s'
)
logger = logging.getLogger("auto_discovery_runner")

def main():
    """Função principal"""
    # Determinar o caminho para main.py
    current_dir = Path(__file__).parent.absolute()
    main_script = current_dir / "main.py"
    
    if not main_script.exists():
        logger.error(f"Arquivo main.py não encontrado em {current_dir}")
        return 1
    
    # Determinar se deve iniciar GUI ou terminal
    use_gui = "--gui" in sys.argv or "-g" in sys.argv
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv
    
    # Preparar argumentos
    args = [sys.executable, str(main_script)]
    
    # Adicionar argumento de GUI se solicitado
    if use_gui:
        args.append("--gui")
        logger.info("Iniciando com interface gráfica")
    else:
        logger.info("Iniciando em modo terminal")
    
    # Adicionar modo de debug se solicitado
    if debug_mode:
        args.append("--debug")
        logger.info("Modo de debug ativado")
    
    # Garantir que descoberta automática de modelos está ativada
    if "--no-auto-discovery" not in sys.argv:
        logger.info("Descoberta automática de modelos ativada")
    
    # Passar outros argumentos recebidos
    for arg in sys.argv[1:]:
        if arg not in ["--gui", "-g", "--debug", "-d"]:
            args.append(arg)
    
    # Executar o script principal
    logger.info(f"Executando: {' '.join(args)}")
    
    try:
        import subprocess
        result = subprocess.run(args, check=True)
        return result.returncode
    except KeyboardInterrupt:
        logger.info("Execução interrompida pelo usuário")
        return 0
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
