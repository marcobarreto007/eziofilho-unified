#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste de Especialistas Phi-2 - EzioFilho_LLMGraph
-------------------------------------------------
Script para testar todos os 12 especialistas Phi-2 no sistema 
Multi-GPU, incluindo suporte para RTX 2060 e GTX 1070.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Phi2ExpertsTest")

# Importar componentes do sistema
from core.phi2_experts import get_phi2_expert, get_available_phi2_experts
from core.multi_gpu_manager import get_multi_gpu_manager
from core.gpu_monitor import get_gpu_monitor

def test_market_experts():
    """Testa os especialistas de mercado"""
    logger.info("=== Testando Especialistas de Mercado ===")
    
    # Dados de teste para análise de mercado
    market_data = {
        "asset": "AAPL",
        "period": "último trimestre",
        "description": """
        A Apple (AAPL) reportou resultados do primeiro trimestre fiscal acima das expectativas, 
        com receita de US$ 119,6 bilhões e lucro por ação de US$ 2,18. As vendas do iPhone, 
        que representam cerca de 50% da receita total, aumentaram 6% ano a ano para US$ 69,7 bilhões. 
        A empresa também anunciou um aumento de US$ 110 bilhões em seu programa de recompra de ações 
        e elevou seu dividendo em 4%.
        
        Tecnicamente, as ações da Apple têm negociado em um canal ascendente nas últimas 8 semanas, 
        encontrando suporte na média móvel de 50 dias e resistência próxima ao nível psicológico de US$ 200.
        O RSI está em 58, mostrando momentum positivo moderado.
        
        Dados macroeconômicos recentes mostram inflação persistente de 3,5% nos EUA e crescimento do PIB de 2,1%.
        O Fed manteve taxas de juros estáveis no último encontro, mas sinalizou que mais dados serão necessários
        antes de iniciar cortes nas taxas.
        """
    }
    
    # Teste do especialista de sentimento
    sentiment_expert = get_phi2_expert("sentiment_analyst")
    sentiment_result = sentiment_expert.analyze(market_data)
    print(f"\nSentiment Expert Result: {json.dumps(sentiment_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista técnico
    technical_expert = get_phi2_expert("technical_analyst")
    technical_result = technical_expert.analyze(market_data)
    print(f"\nTechnical Expert Result: {json.dumps(technical_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista fundamental
    fundamental_expert = get_phi2_expert("fundamental_analyst")
    fundamental_result = fundamental_expert.analyze(market_data)
    print(f"\nFundamental Expert Result: {json.dumps(fundamental_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista macroeconômico
    macro_expert = get_phi2_expert("macro_economist")
    macro_result = macro_expert.analyze(market_data)
    print(f"\nMacro Expert Result: {json.dumps(macro_result, indent=2, ensure_ascii=False)}\n")
    
    return {
        "sentiment": sentiment_result,
        "technical": technical_result,
        "fundamental": fundamental_result,
        "macro": macro_result
    }

def test_risk_experts():
    """Testa os especialistas de risco"""
    logger.info("=== Testando Especialistas de Risco ===")
    
    # Dados de teste para análise de risco
    risk_data = {
        "portfolio": "Portfólio Global Diversificado",
        "period": "atual",
        "description": """
        O portfólio global diversificado possui uma alocação de 60% em ações (35% EUA, 15% Europa, 10% mercados emergentes),
        30% em renda fixa (15% títulos governamentais, 10% corporativos investment grade, 5% high yield) e 10% em alternativos
        (5% ouro, 3% commodities, 2% criptomoedas).
        
        A volatilidade anualizada do portfólio está em 14%, com drawdown máximo de 8% nos últimos 12 meses.
        O VaR diário (95%) é de 1,2% e o beta em relação ao S&P 500 é de 0,78.
        
        As taxas de juros dos EUA estão em 5,25%, com expectativa de cortes graduais nos próximos 12 meses.
        A curva de juros está ligeiramente invertida com spreads de crédito em níveis historicamente baixos.
        
        Os ratings médios da parcela de crédito são 'BBB+' para corporativos e 'BB-' para high yield.
        A duration média da carteira de renda fixa é de 5,8 anos.
        
        A liquidez do portfólio permite liquidação de 85% dos ativos em até 2 dias sem impacto significativo de mercado.
        """
    }
    
    # Teste do especialista de gerenciamento de risco
    risk_manager = get_phi2_expert("risk_manager")
    risk_manager_result = risk_manager.analyze(risk_data)
    print(f"\nRisk Manager Result: {json.dumps(risk_manager_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista de volatilidade
    volatility_expert = get_phi2_expert("volatility_expert") 
    volatility_result = volatility_expert.analyze(risk_data)
    print(f"\nVolatility Expert Result: {json.dumps(volatility_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista de crédito
    credit_expert = get_phi2_expert("credit_analyst")
    credit_result = credit_expert.analyze(risk_data)
    print(f"\nCredit Expert Result: {json.dumps(credit_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista de liquidez
    liquidity_expert = get_phi2_expert("liquidity_specialist")
    liquidity_result = liquidity_expert.analyze(risk_data)
    print(f"\nLiquidity Expert Result: {json.dumps(liquidity_result, indent=2, ensure_ascii=False)}\n")
    
    return {
        "risk_manager": risk_manager_result,
        "volatility": volatility_result,
        "credit": credit_result,
        "liquidity": liquidity_result
    }

def test_quant_experts():
    """Testa os especialistas quantitativos"""
    logger.info("=== Testando Especialistas Quantitativos ===")
    
    # Dados de teste para análise quantitativa
    quant_data = {
        "strategy": "Estratégia Quantitativa Multi-ativos",
        "period": "atual",
        "description": """
        A estratégia quantitativa multi-ativos emprega uma combinação de modelos de momentum, mean-reversion e 
        carry trade em diferentes classes de ativos, incluindo ações, renda fixa, moedas e commodities.
        
        O algoritmo principal utiliza uma janela de lookback de 63 dias para momentum, ponderando sinais com
        volatilidade realizada. Os trades de mean-reversion são acionados por 2 desvios-padrão do Z-score calculado
        sobre uma média móvel de 21 dias. O modelo de carry utiliza diferencial de taxas de juros para FX e curvas
        de commodities para sinais de roll-yield.
        
        O mercado de opções atualmente mostra um skew pronunciado em índices de ações, com volatilidade implícita
        de 22% nos strikes ATM do S&P 500 para vencimentos em 30 dias. A estrutura a termo está em contango,
        com diferencial de 3% entre contratos de 30 e 90 dias.
        
        A curva de juros dos EUA está ligeiramente invertida, com spreads de 20 bps entre 2 e 10 anos.
        Títulos corporativos investment grade oferecem spreads de 125 bps sobre Treasuries, enquanto high yield
        está com spreads médios de 380 bps.
        
        O mercado de criptomoedas mostra dominância de Bitcoin em 55%, com volume de transações on-chain crescendo
        15% no último trimestre. A volatilidade realizada está em 45% anualizada, com funding rates positivos em
        contratos perpétuos, indicando posicionamento líquido comprado.
        """
    }
    
    # Teste do especialista algorítmico
    algo_expert = get_phi2_expert("algorithmic_trader")
    algo_result = algo_expert.analyze(quant_data)
    print(f"\nAlgorithmic Expert Result: {json.dumps(algo_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista de opções
    options_expert = get_phi2_expert("options_specialist")
    options_result = options_expert.analyze(quant_data)
    print(f"\nOptions Expert Result: {json.dumps(options_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista de renda fixa
    fixed_income_expert = get_phi2_expert("fixed_income")
    fixed_income_result = fixed_income_expert.analyze(quant_data)
    print(f"\nFixed Income Expert Result: {json.dumps(fixed_income_result, indent=2, ensure_ascii=False)}\n")
    
    # Teste do especialista de cripto
    crypto_expert = get_phi2_expert("crypto_analyst")
    crypto_result = crypto_expert.analyze(quant_data)
    print(f"\nCrypto Expert Result: {json.dumps(crypto_result, indent=2, ensure_ascii=False)}\n")
    
    return {
        "algorithmic": algo_result,
        "options": options_result,
        "fixed_income": fixed_income_result,
        "crypto": crypto_result
    }

def run_multi_gpu_test(output_file: Optional[str] = None):
    """
    Executa o teste de todos os especialistas Phi-2 no sistema Multi-GPU
    
    Args:
        output_file: Caminho para arquivo de saída dos resultados (opcional)
    """
    # Iniciar gerenciador Multi-GPU e monitor
    gpu_manager = get_multi_gpu_manager()
    gpu_monitor = get_gpu_monitor()
    
    # Verificar GPUs disponíveis
    try:
        gpus_info = gpu_monitor.get_gpu_info()
        logger.info(f"GPUs Disponíveis: {len(gpus_info)}")
        for i, gpu in enumerate(gpus_info):
            logger.info(f"GPU {i}: {gpu['name']} - Memória: {gpu['memory_total']}MB")
    except Exception as e:
        logger.warning(f"Não foi possível obter informações das GPUs: {e}")
        logger.warning("Prosseguindo sem informações detalhadas de GPU")
    
    # Iniciar teste
    start_time = time.time()
    logger.info("Iniciando teste de especialistas Phi-2...")
    
    # Testar especialistas de cada categoria
    results = {}
    
    try:
        # Testar especialistas de mercado
        market_results = test_market_experts()
        results["market_experts"] = market_results
        
        # Testar especialistas de risco
        risk_results = test_risk_experts()
        results["risk_experts"] = risk_results
        
        # Testar especialistas quantitativos
        quant_results = test_quant_experts()
        results["quant_experts"] = quant_results
        
    except Exception as e:
        logger.error(f"Erro durante os testes: {e}")
        results["error"] = str(e)
    
    # Finalizar teste
    elapsed_time = time.time() - start_time
    logger.info(f"Teste concluído em {elapsed_time:.2f} segundos")
    
    # Obter estatísticas de uso das GPUs
    try:
        gpu_stats = gpu_monitor.get_gpu_info()
        results["gpu_statistics"] = gpu_stats
    except Exception as e:
        logger.warning(f"Não foi possível obter estatísticas de GPU: {e}")
        results["gpu_statistics"] = []
    
    results["total_time"] = elapsed_time
    results["timestamp"] = time.time()
    
    # Salvar resultados em arquivo se solicitado
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Resultados salvos em {output_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar resultados: {e}")
    
    return results

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Teste de Especialistas Phi-2 - EzioFilho_LLMGraph")
    parser.add_argument('--output', '-o', type=str, help='Arquivo de saída para os resultados (JSON)')
    parser.add_argument('--category', '-c', type=str, choices=['market', 'risk', 'quant', 'all'], 
                        default='all', help='Categoria de especialistas para testar')
    args = parser.parse_args()
    
    # Configurar caminho de saída padrão se não fornecido
    output_file = args.output
    if not output_file:
        output_dir = Path("./results")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"phi2_experts_test_{int(time.time())}.json"
    
    # Executar teste de acordo com a categoria escolhida
    if args.category == 'market':
        results = {"market_experts": test_market_experts()}
    elif args.category == 'risk':
        results = {"risk_experts": test_risk_experts()}
    elif args.category == 'quant':
        results = {"quant_experts": test_quant_experts()}
    else:  # 'all'
        results = run_multi_gpu_test(str(output_file))
    
    # Resumir resultados
    logger.info("=== Resumo dos Resultados ===")
    for category, category_results in results.items():
        if category in ["gpu_statistics", "total_time", "timestamp", "error"]:
            continue
            
        logger.info(f"Categoria: {category}")
        for expert_name, expert_result in category_results.items():
            confidence = expert_result.get("confidence", "N/A")
            processing_time = expert_result.get("processing_time", "N/A")
            logger.info(f"  - {expert_name}: Confiança={confidence}, Tempo={processing_time:.2f}s")
    
    if "total_time" in results:
        logger.info(f"Tempo total de execução: {results['total_time']:.2f} segundos")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
