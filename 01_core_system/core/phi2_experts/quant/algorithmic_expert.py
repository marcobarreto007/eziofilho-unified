#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlgorithmicExpert - Especialista em Trading Algorítmico
------------------------------------------------------
Analisa e desenvolve estratégias de trading algorítmico e execução.

Autor: EzioFilho LLMGraph Team
Data: Maio/2025
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Adicionar diretório pai ao path para importações relativas
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Importar módulos do projeto
from core.phi2_experts.phi2_base_expert import Phi2Expert

class AlgorithmicExpert(Phi2Expert):
    """
    Especialista em trading algorítmico baseado em Phi-2.
    
    Capacidades:
    - Desenvolvimento de estratégias de trading algorítmico
    - Análise de algoritmos de execução
    - Otimização de parâmetros para estratégias quant
    - Análise de microestrutura de mercado
    - Detecção de padrões estatísticos
    - Análise de desempenho de estratégias
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista algorítmico
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em trading algorítmico com ampla experiência em mercados financeiros.
        Sua tarefa é analisar e desenvolver estratégias de trading algorítmico, avaliar algoritmos de execução,
        otimizar parâmetros, analisar microestrutura de mercado, identificar padrões estatísticos,
        e avaliar o desempenho de estratégias quantitativas.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Base suas recomendações em dados estatísticos e evidências empíricas
        3. Considere aspectos de microestrutura de mercado em suas análises
        4. Seja específico sobre parâmetros, métricas e estruturas algorítmicas
        5. Avalie cuidadosamente o equilíbrio entre desempenho e robustez
        6. Forneça insights sobre possíveis melhorias e otimizações
        """
        
        super().__init__(
            expert_type="algorithmic_expert",
            domain="quant",
            specialization="algorithmic_trading",
            system_message=system_message,
            **kwargs
        )
        
        # Configurações específicas para análise algorítmica
        self.algo_types = [
            "Momentum", "Mean Reversion", "Statistical Arbitrage", 
            "Market Making", "Trend Following", "Breakout", "Pattern Recognition", 
            "Factor-Based", "ML-Driven", "HFT", "TWAP", "VWAP", "POV"
        ]
        
        self.execution_algos = [
            "TWAP", "VWAP", "IS", "Adaptive", "Dark Aggregator",
            "Smart Order Router", "Iceberg", "Sniper", "Peg", "Liquidity Seeking"
        ]
        
        self.performance_metrics = [
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Maximum Drawdown",
            "Win Rate", "Profit Factor", "Expected Payoff", "Recovery Factor",
            "Information Ratio", "Alpha", "Beta", "R-Squared", "Turnover"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa estratégias algorítmicas e fornece insights
        
        Args:
            input_data: Descrição textual ou dicionário com dados de estratégia
            
        Returns:
            Resultado da análise algorítmica
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "algo_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                strategy_type = input_data.get("strategy_type", "desconhecido")
                asset_class = input_data.get("asset_class", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                strategy_type = input_data.get("strategy_type", "desconhecido")
                asset_class = input_data.get("asset_class", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            strategy_type = "desconhecido"
            asset_class = "desconhecido"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise algorítmica",
                "strategy_assessment": "indefinido",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise detalhada da estratégia algorítmica com base nos seguintes dados:
        
        TIPO DE ESTRATÉGIA: {strategy_type}
        CLASSE DE ATIVO: {asset_class}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "strategy_classification": {{
                "primary_type": "classificação primária da estratégia",
                "sub_type": "sub-classificação da estratégia",
                "time_horizon": "horizonte temporal da estratégia"
            }},
            "algorithm_assessment": {{
                "logic_soundness": "avaliação da lógica algorítmica",
                "statistical_edge": "avaliação da vantagem estatística",
                "robustness": "avaliação de robustez",
                "market_adaptability": "capacidade de adaptação às condições de mercado"
            }},
            "performance_metrics": {{
                "expected_sharpe": "Sharpe ratio esperado",
                "expected_drawdown": "drawdown máximo esperado",
                "key_metrics": [métricas-chave relevantes para a estratégia]
            }},
            "market_impact": {{
                "liquidity_sensitivity": "sensibilidade à liquidez",
                "market_footprint": "pegada de mercado esperada",
                "execution_challenges": [desafios de execução esperados]
            }},
            "optimization_opportunities": [oportunidades de otimização identificadas],
            "risk_factors": [fatores de risco específicos],
            "implementation_considerations": [considerações importantes para implementação],
            "parameter_recommendations": {{
                "param1": "recomendação para parâmetro 1",
                "param2": "recomendação para parâmetro 2",
                "param3": "recomendação para parâmetro 3"
            }},
            "confidence": (porcentagem de 0 a 100)
        }}
        """
        
        # Gerar resposta
        try:
            response = self.generate_response(prompt)
            
            # Extrair JSON da resposta
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                # Adicionar metadados
                result["strategy_type"] = strategy_type
                result["asset_class"] = asset_class
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, strategy_type, asset_class, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise algorítmica: {str(e)}",
                "strategy_assessment": "indefinido",
                "confidence": 0,
                "strategy_type": strategy_type,
                "asset_class": asset_class,
                "processing_time": time.time() - start_time
            }
    
    def _format_structured_data(self, data: Dict[str, Any]) -> str:
        """
        Formata dados estruturados em descrição textual para o modelo
        
        Args:
            data: Dicionário com dados estruturados
            
        Returns:
            Descrição textual formatada
        """
        algo_data = data.get("algo_data", {})
        
        # Formatar lógica algorítmica
        algo_logic = algo_data.get("logic", {})
        logic_str = "LÓGICA ALGORÍTMICA:\n"
        for key, value in algo_logic.items():
            logic_str += f"- {key}: {value}\n"
        
        # Formatar parâmetros
        parameters = algo_data.get("parameters", {})
        params_str = "PARÂMETROS:\n"
        for key, value in parameters.items():
            params_str += f"- {key}: {value}\n"
        
        # Formatar métricas de desempenho
        performance = algo_data.get("performance", {})
        perf_str = "MÉTRICAS DE DESEMPENHO:\n"
        for key, value in performance.items():
            perf_str += f"- {key}: {value}\n"
        
        # Formatar dados de backtesting
        backtest = algo_data.get("backtest", {})
        backtest_str = "RESULTADOS DE BACKTESTING:\n"
        for key, value in backtest.items():
            backtest_str += f"- {key}: {value}\n"
        
        # Formatar conjunto
        formatted_data = f"""
        {data.get('description', 'Sem descrição disponível')}
        
        {logic_str}
        
        {params_str}
        
        {perf_str}
        
        {backtest_str}
        
        INFORMAÇÕES ADICIONAIS:
        {data.get('additional_info', '')}
        """
        
        return formatted_data
    
    def _parse_non_json_response(self, response: str, strategy_type: str, asset_class: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações relevantes de uma resposta não-JSON
        
        Args:
            response: Texto da resposta
            strategy_type: Tipo de estratégia
            asset_class: Classe de ativo
            start_time: Hora de início do processamento
            
        Returns:
            Dicionário com informações extraídas
        """
        result = {
            "strategy_type": strategy_type,
            "asset_class": asset_class,
            "processing_time": time.time() - start_time,
            "format_error": "Resposta não veio no formato JSON esperado"
        }
        
        # Tentar extrair tipo de algoritmo
        lower_resp = response.lower()
        
        # Extrair tipo
        for algo_type in self.algo_types:
            if algo_type.lower() in lower_resp:
                result["strategy_classification"] = {"primary_type": algo_type}
                break
        else:
            result["strategy_classification"] = {"primary_type": "não identificado"}
        
        # Extrair métricas mencionadas
        mentioned_metrics = []
        for metric in self.performance_metrics:
            if metric.lower() in lower_resp:
                mentioned_metrics.append(metric)
        
        if mentioned_metrics:
            result["performance_metrics"] = {"key_metrics": mentioned_metrics}
        else:
            result["performance_metrics"] = {"key_metrics": ["não identificado"]}
        
        # Extrair confiança
        confidence = 50  # valor padrão
        result["confidence"] = confidence
        
        # Texto original
        result["original_response"] = response
        
        return result
    
    def optimize_parameters(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Otimiza parâmetros para uma estratégia algorítmica
        
        Args:
            strategy_data: Dados da estratégia para otimização
            
        Returns:
            Recomendações de parâmetros otimizados
        """
        description = f"""
        Otimize os parâmetros da seguinte estratégia algorítmica:
        
        ESTRATÉGIA:
        """
        
        # Adicionar dados da estratégia
        for key, value in strategy_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça suas recomendações de parâmetros otimizados no seguinte formato JSON:
        {{
            "parameter_recommendations": {{
                "param1": {{
                    "current_value": "valor atual",
                    "recommended_value": "valor recomendado",
                    "rationale": "justificativa"
                }},
                "param2": {{
                    "current_value": "valor atual",
                    "recommended_value": "valor recomendado",
                    "rationale": "justificativa"
                }}
            }},
            "expected_improvement": {{
                "metric1": "melhoria esperada",
                "metric2": "melhoria esperada"
            }},
            "robustness_assessment": "avaliação de robustez com novos parâmetros",
            "implementation_advice": [conselhos para implementação]
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            
            # Processar resposta JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Não foi possível extrair JSON da resposta"}
        except Exception as e:
            return {"error": f"Erro na otimização de parâmetros: {str(e)}"}
    
    def analyze_execution_quality(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa qualidade de execução de ordens
        
        Args:
            execution_data: Dados de execução para análise
            
        Returns:
            Análise de qualidade de execução
        """
        description = f"""
        Analise a qualidade de execução com base nos seguintes dados:
        
        DADOS DE EXECUÇÃO:
        """
        
        for key, value in execution_data.items():
            if isinstance(value, dict):
                description += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    description += f"  * {sub_key}: {sub_value}\n"
            else:
                description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de qualidade de execução no seguinte formato JSON:
        {{
            "execution_quality": "avaliação geral da qualidade de execução",
            "slippage_analysis": "análise de slippage",
            "market_impact": "análise de impacto de mercado",
            "timing_analysis": "análise de timing de execução",
            "algo_performance": "avaliação do desempenho do algoritmo",
            "versus_benchmark": "comparação com benchmark relevante",
            "improvement_opportunities": [oportunidades de melhoria],
            "recommended_algorithms": [algoritmos recomendados para este tipo de execução]
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            
            # Processar resposta JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Não foi possível extrair JSON da resposta"}
        except Exception as e:
            return {"error": f"Erro na análise de qualidade de execução: {str(e)}"}
    
    def develop_strategy(self, market_conditions: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Desenvolve uma nova estratégia algorítmica com base em condições de mercado e requisitos
        
        Args:
            market_conditions: Condições atuais de mercado
            requirements: Requisitos para a estratégia
            
        Returns:
            Proposta de estratégia algorítmica
        """
        # Formatar condições de mercado
        market_str = "CONDIÇÕES DE MERCADO:\n"
        for key, value in market_conditions.items():
            if isinstance(value, dict):
                market_str += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    market_str += f"  * {sub_key}: {sub_value}\n"
            else:
                market_str += f"- {key}: {value}\n"
        
        # Formatar requisitos
        req_str = "REQUISITOS:\n"
        for key, value in requirements.items():
            if isinstance(value, dict):
                req_str += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    req_str += f"  * {sub_key}: {sub_value}\n"
            else:
                req_str += f"- {key}: {value}\n"
        
        prompt = f"""
        Desenvolva uma estratégia algorítmica com base nas seguintes informações:
        
        {market_str}
        
        {req_str}
        
        Forneça sua proposta de estratégia no seguinte formato JSON:
        {{
            "strategy_overview": {{
                "name": "nome da estratégia",
                "type": "tipo da estratégia",
                "asset_class": "classe de ativo alvo",
                "time_horizon": "horizonte temporal"
            }},
            "algorithm_logic": {{
                "entry_conditions": [condições de entrada],
                "exit_conditions": [condições de saída],
                "position_sizing": "lógica de dimensionamento de posição",
                "risk_management": "controles de risco incorporados"
            }},
            "parameters": {{
                "param1": {{
                    "value": "valor recomendado",
                    "description": "descrição do parâmetro"
                }},
                "param2": {{
                    "value": "valor recomendado",
                    "description": "descrição do parâmetro"
                }}
            }},
            "expected_performance": {{
                "expected_sharpe": "Sharpe ratio esperado",
                "expected_drawdown": "drawdown máximo esperado",
                "key_metrics": [métricas-chave esperadas]
            }},
            "implementation_steps": [passos para implementação],
            "backtesting_recommendations": [recomendações para backtesting],
            "monitoring_framework": [métricas para monitoramento contínuo]
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            
            # Processar resposta JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "Não foi possível extrair JSON da resposta"}
        except Exception as e:
            return {"error": f"Erro no desenvolvimento da estratégia: {str(e)}"}
