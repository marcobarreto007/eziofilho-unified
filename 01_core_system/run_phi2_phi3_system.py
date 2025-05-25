#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EzioFilho_LLMGraph - Sistema Completo
-------------------------------------
Script principal para iniciar o sistema completo,
integrando especialistas Phi-2 e o cérebro central Phi-3.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("EzioFilho_LLMGraph")

# Importar componentes do sistema
from core.phi2_phi3_integration import get_phi2_phi3_integrator
from core.gpu_monitor import get_gpu_monitor
from core.multi_gpu_manager import get_multi_gpu_manager

def init_system(config_path=None):
    """
    Inicializa o sistema completo.
    
    Args:
        config_path: Caminho para arquivo de configuração (opcional)
    """
    logger.info("Inicializando EzioFilho_LLMGraph...")
    
    # Carregar configuração
    config = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuração carregada de {config_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
    
    # Inicializar monitor de GPU
    gpu_monitor = get_gpu_monitor()
    
    # Verificar GPUs disponíveis
    try:
        gpus_info = gpu_monitor.get_gpu_info()
        logger.info(f"GPUs Disponíveis: {len(gpus_info)}")
        for i, gpu in enumerate(gpus_info):
            logger.info(f"GPU {i}: {gpu['name']} - Memória: {gpu['memory_total']}MB")
    except Exception as e:
        logger.warning(f"Não foi possível obter informações das GPUs: {e}")
    
    # Inicializar gerenciador Multi-GPU
    gpu_manager = get_multi_gpu_manager()
    
    # Inicializar integrador Phi-2/Phi-3
    integrator = get_phi2_phi3_integrator()
    
    logger.info("Sistema EzioFilho_LLMGraph inicializado com sucesso!")
    return integrator

def run_market_analysis(integrator, market_data):
    """
    Executa análise de mercado.
    
    Args:
        integrator: Instância do integrador
        market_data: Dados de mercado para análise
    """
    logger.info(f"Iniciando análise de mercado para {market_data.get('asset', 'ativo')}")
    
    # Especialistas de mercado
    market_experts = [
        "sentiment_analyst", 
        "technical_analyst", 
        "fundamental_analyst", 
        "macro_economist"
    ]
    
    # Executar análise
    start_time = time.time()
    result = integrator.analyze_with_full_system(market_data, market_experts)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Análise de mercado concluída em {elapsed_time:.2f}s")
    return result

def run_risk_analysis(integrator, risk_data):
    """
    Executa análise de risco.
    
    Args:
        integrator: Instância do integrador
        risk_data: Dados de risco para análise
    """
    logger.info(f"Iniciando análise de risco para {risk_data.get('portfolio', 'portfólio')}")
    
    # Especialistas de risco
    risk_experts = [
        "risk_manager", 
        "volatility_expert", 
        "credit_analyst", 
        "liquidity_specialist"
    ]
    
    # Executar análise
    start_time = time.time()
    result = integrator.analyze_with_full_system(risk_data, risk_experts)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Análise de risco concluída em {elapsed_time:.2f}s")
    return result

def run_quant_analysis(integrator, quant_data):
    """
    Executa análise quantitativa.
    
    Args:
        integrator: Instância do integrador
        quant_data: Dados quantitativos para análise
    """
    logger.info(f"Iniciando análise quantitativa para {quant_data.get('strategy', 'estratégia')}")
    
    # Especialistas quantitativos
    quant_experts = [
        "algorithmic_trader", 
        "options_specialist", 
        "fixed_income", 
        "crypto_analyst"
    ]
    
    # Executar análise
    start_time = time.time()
    result = integrator.analyze_with_full_system(quant_data, quant_experts)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Análise quantitativa concluída em {elapsed_time:.2f}s")
    return result

def run_full_analysis(integrator, data):
    """
    Executa análise completa com todos os especialistas.
    
    Args:
        integrator: Instância do integrador
        data: Dados para análise
    """
    logger.info(f"Iniciando análise completa para {data.get('subject', 'consulta')}")
    
    # Executar análise com todos os especialistas
    start_time = time.time()
    result = integrator.analyze_with_full_system(data)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Análise completa concluída em {elapsed_time:.2f}s")
    return result

def console_interface(integrator):
    """
    Executa interface de console para o sistema.
    
    Args:
        integrator: Instância do integrador
    """
    logger.info("Iniciando interface de console EzioFilho_LLMGraph")
    print("\n=== EzioFilho_LLMGraph - Sistema de Análise Financeira ===\n")
    
    while True:
        print("\nEscolha uma opção:")
        print("1. Análise de Mercado")
        print("2. Análise de Risco")
        print("3. Análise Quantitativa")
        print("4. Análise Completa")
        print("5. Sair")
        
        choice = input("\nOpção: ")
        
        if choice == "1":
            asset = input("\nAtivo (ex: AAPL): ")
            period = input("Período (ex: Q1 2025): ")
            description = input("Descrição (pressione Enter para finalizar):\n")
            
            market_data = {
                "asset": asset,
                "period": period,
                "description": description
            }
            
            result = run_market_analysis(integrator, market_data)
            
            # Salvar resultado
            output_dir = Path("./results")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"market_analysis_{asset.lower()}_{int(time.time())}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            print(f"\nAnálise concluída e salva em {output_file}")
            
        elif choice == "2":
            portfolio = input("\nPortfólio: ")
            period = input("Período: ")
            description = input("Descrição (pressione Enter para finalizar):\n")
            
            risk_data = {
                "portfolio": portfolio,
                "period": period,
                "description": description
            }
            
            result = run_risk_analysis(integrator, risk_data)
            
            # Salvar resultado
            output_dir = Path("./results")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"risk_analysis_{int(time.time())}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            print(f"\nAnálise concluída e salva em {output_file}")
            
        elif choice == "3":
            strategy = input("\nEstratégia: ")
            period = input("Período: ")
            description = input("Descrição (pressione Enter para finalizar):\n")
            
            quant_data = {
                "strategy": strategy,
                "period": period,
                "description": description
            }
            
            result = run_quant_analysis(integrator, quant_data)
            
            # Salvar resultado
            output_dir = Path("./results")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"quant_analysis_{int(time.time())}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            print(f"\nAnálise concluída e salva em {output_file}")
            
        elif choice == "4":
            subject = input("\nAssunto: ")
            period = input("Período: ")
            description = input("Descrição (pressione Enter para finalizar):\n")
            
            full_data = {
                "subject": subject,
                "period": period,
                "description": description
            }
            
            result = run_full_analysis(integrator, full_data)
            
            # Salvar resultado
            output_dir = Path("./results")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"full_analysis_{int(time.time())}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            print(f"\nAnálise completa concluída e salva em {output_file}")
            
        elif choice == "5":
            print("\nEncerrando sistema EzioFilho_LLMGraph...")
            break
            
        else:
            print("\nOpção inválida.")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="EzioFilho_LLMGraph - Sistema Financeiro com IA")
    parser.add_argument('--config', '-c', type=str, help='Arquivo de configuração')
    parser.add_argument('--multi-gpu', '-m', action='store_true', help='Ativar modo Multi-GPU')
    parser.add_argument('--console', action='store_true', help='Iniciar interface de console')
    parser.add_argument('--market', type=str, help='Arquivo JSON com dados para análise de mercado')
    parser.add_argument('--risk', type=str, help='Arquivo JSON com dados para análise de risco')
    parser.add_argument('--quant', type=str, help='Arquivo JSON com dados para análise quantitativa')
    parser.add_argument('--output', '-o', type=str, help='Diretório para salvar resultados')
    args = parser.parse_args()
    
    # Configurar diretório de saída
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar sistema
    integrator = init_system(args.config)
    
    # Executar análise a partir de arquivos JSON
    if args.market:
        try:
            with open(args.market, 'r', encoding='utf-8') as f:
                market_data = json.load(f)
            result = run_market_analysis(integrator, market_data)
            
            output_file = output_dir / f"market_analysis_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Análise de mercado salva em {output_file}")
        except Exception as e:
            logger.error(f"Erro ao processar arquivo de análise de mercado: {e}")
    
    if args.risk:
        try:
            with open(args.risk, 'r', encoding='utf-8') as f:
                risk_data = json.load(f)
            result = run_risk_analysis(integrator, risk_data)
            
            output_file = output_dir / f"risk_analysis_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Análise de risco salva em {output_file}")
        except Exception as e:
            logger.error(f"Erro ao processar arquivo de análise de risco: {e}")
    
    if args.quant:
        try:
            with open(args.quant, 'r', encoding='utf-8') as f:
                quant_data = json.load(f)
            result = run_quant_analysis(integrator, quant_data)
            
            output_file = output_dir / f"quant_analysis_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Análise quantitativa salva em {output_file}")
        except Exception as e:
            logger.error(f"Erro ao processar arquivo de análise quantitativa: {e}")
    
    # Iniciar interface de console se solicitado
    if args.console or (not args.market and not args.risk and not args.quant):
        console_interface(integrator)
    
    logger.info("Execução concluída.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
