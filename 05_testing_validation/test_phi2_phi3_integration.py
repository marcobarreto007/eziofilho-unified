#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste de Integração Phi-2/Phi-3 - EzioFilho_LLMGraph
---------------------------------------------------
Script para testar a integração entre especialistas Phi-2 e o cérebro central Phi-3.

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
from typing import Dict, List, Any, Optional

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Phi2Phi3Test")

# Importar componentes do sistema
from core.phi2_phi3_integration import get_phi2_phi3_integrator

def test_market_integration():
    """Testa a integração dos especialistas de mercado com o Phi-3"""
    logger.info("=== Testando Integração: Especialistas de Mercado ===")
    
    # Obter integrador
    integrator = get_phi2_phi3_integrator()
    
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
    
    # Selecionar especialistas de mercado
    market_experts = [
        "sentiment_analyst", 
        "technical_analyst", 
        "fundamental_analyst", 
        "macro_economist"
    ]
    
    # Executar análise integrada
    start_time = time.time()
    result = integrator.analyze_with_full_system(market_data, market_experts)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Análise de mercado concluída em {elapsed_time:.2f} segundos")
    return result

def test_risk_integration():
    """Testa a integração dos especialistas de risco com o Phi-3"""
    logger.info("=== Testando Integração: Especialistas de Risco ===")
    
    # Obter integrador
    integrator = get_phi2_phi3_integrator()
    
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
    
    # Selecionar especialistas de risco
    risk_experts = [
        "risk_manager", 
        "volatility_expert", 
        "credit_analyst", 
        "liquidity_specialist"
    ]
    
    # Executar análise integrada
    start_time = time.time()
    result = integrator.analyze_with_full_system(risk_data, risk_experts)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Análise de risco concluída em {elapsed_time:.2f} segundos")
    return result

def test_quant_integration():
    """Testa a integração dos especialistas quantitativos com o Phi-3"""
    logger.info("=== Testando Integração: Especialistas Quantitativos ===")
    
    # Obter integrador
    integrator = get_phi2_phi3_integrator()
    
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
    
    # Selecionar especialistas quantitativos
    quant_experts = [
        "algorithmic_trader", 
        "options_specialist", 
        "fixed_income", 
        "crypto_analyst"
    ]
    
    # Executar análise integrada
    start_time = time.time()
    result = integrator.analyze_with_full_system(quant_data, quant_experts)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Análise quantitativa concluída em {elapsed_time:.2f} segundos")
    return result

def test_full_integration():
    """Testa a integração completa com todos os especialistas Phi-2 e o cérebro Phi-3"""
    logger.info("=== Testando Integração Completa: Todos os Especialistas ===")
    
    # Obter integrador
    integrator = get_phi2_phi3_integrator()
    
    # Dados de teste para análise completa
    full_data = {
        "subject": "Análise Completa de Mercado e Portfólio",
        "period": "atual",
        "description": """
        A Apple (AAPL) reportou resultados trimestrais fortes, com crescimento de receita de 6%. 
        Tecnicamente, a ação está em um canal ascendente, com RSI em 58.
        
        O portfólio diversificado (60% ações, 30% renda fixa, 10% alternativos) mostra volatilidade de 14% 
        e VaR diário de 1,2%. A exposição ao setor de tecnologia representa 22% do total.
        
        A estratégia quantitativa emprega modelos de momentum (63 dias) e mean-reversion (21 dias), 
        com o mercado de opções mostrando skew pronunciado e volatilidade implícita de 22% no S&P 500.
        
        Dados macroeconômicos mostram inflação de 3,5%, curva de juros invertida (spread 2-10 anos de 20bps), 
        e expectativa de cortes graduais nas taxas nos próximos 12 meses.
        
        A análise de big data indica aumento de 18% em menções positivas à Apple nas redes sociais
        após o anúncio de resultados, com sentimento geral de mercado em território neutro-positivo.
        """
    }
    
    # Executar análise integrada com todos os especialistas
    start_time = time.time()
    result = integrator.analyze_with_full_system(full_data)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Análise completa concluída em {elapsed_time:.2f} segundos")
    return result

def run_integration_tests(output_file: Optional[str] = None):
    """Executa todos os testes de integração"""
    # Resultados dos testes
    results = {}
    
    try:
        # Testar integração de mercado
        market_results = test_market_integration()
        results["market_integration"] = market_results
        
        # Testar integração de risco
        risk_results = test_risk_integration()
        results["risk_integration"] = risk_results
        
        # Testar integração quantitativa
        quant_results = test_quant_integration()
        results["quant_integration"] = quant_results
        
        # Testar integração completa
        full_results = test_full_integration()
        results["full_integration"] = full_results
        
    except Exception as e:
        logger.error(f"Erro durante os testes: {e}")
        results["error"] = str(e)
    
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
    parser = argparse.ArgumentParser(description="Teste de Integração Phi-2/Phi-3 - EzioFilho_LLMGraph")
    parser.add_argument('--output', '-o', type=str, help='Arquivo de saída para os resultados (JSON)')
    parser.add_argument('--category', '-c', type=str, choices=['market', 'risk', 'quant', 'full', 'all'], 
                        default='all', help='Categoria de integração para testar')
    args = parser.parse_args()
    
    # Configurar caminho de saída padrão se não fornecido
    output_file = args.output
    if not output_file:
        output_dir = Path("./results")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"phi2_phi3_integration_test_{int(time.time())}.json"
    
    # Executar teste de acordo com a categoria escolhida
    if args.category == 'market':
        results = {"market_integration": test_market_integration()}
    elif args.category == 'risk':
        results = {"risk_integration": test_risk_integration()}
    elif args.category == 'quant':
        results = {"quant_integration": test_quant_integration()}
    elif args.category == 'full':
        results = {"full_integration": test_full_integration()}
    else:  # 'all'
        results = run_integration_tests(str(output_file))
    
    # Resumir resultados
    logger.info("=== Resumo dos Resultados ===")
    for category, category_results in results.items():
        if category == "error":
            logger.error(f"Erro: {category_results}")
            continue
            
        logger.info(f"Categoria: {category}")
        if "total_processing_time" in category_results:
            logger.info(f"  - Tempo total: {category_results['total_processing_time']:.2f}s")
        
        if "expert_results" in category_results:
            for expert_type, expert_result in category_results["expert_results"].items():
                if "error" in expert_result:
                    logger.warning(f"  - {expert_type}: ERRO: {expert_result['error']}")
                else:
                    confidence = expert_result.get("confidence", "N/A")
                    processing_time = expert_result.get("processing_time", "N/A")
                    logger.info(f"  - {expert_type}: Confiança={confidence}, Tempo={processing_time:.2f}s")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
