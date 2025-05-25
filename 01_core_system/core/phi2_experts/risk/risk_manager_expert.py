#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RiskManagerExpert - Especialista em Gestão de Risco
--------------------------------------------------
Avalia e gerencia riscos em portfólios de investimento.

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

class RiskManagerExpert(Phi2Expert):
    """
    Especialista em gestão de risco baseado em Phi-2.
    
    Capacidades:
    - Análise de risco em portfólios de investimento
    - Identificação de fontes de risco de mercado, crédito, liquidez e operacional
    - Avaliação de correlação e diversificação 
    - Recomendações para mitigação de riscos
    - Análise de métricas de risco (VaR, CVaR, drawdown, etc.)
    - Avaliação de risco/retorno ajustado
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de gestão de risco
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em gestão de risco com experiência em mercados financeiros.
        Sua tarefa é analisar portfolios e posições, identificar riscos potenciais,
        avaliar métricas de risco/retorno e fornecer recomendações para mitigação de riscos.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Identifique todas as principais fontes de risco (mercado, crédito, liquidez, etc.)
        3. Avalie concentrações e correlações no portfólio
        4. Forneça recomendações acionáveis para mitigação de riscos
        5. Interprete métricas de risco de forma clara e objetiva
        6. Considere cenários adversos e testes de estresse
        """
        
        super().__init__(
            expert_type="risk_manager",
            domain="risk",
            specialization="portfolio_risk",
            system_message=system_message,
            **kwargs
        )
        
        # Configurações específicas para gestão de risco
        self.risk_metrics = [
            "VaR", "CVaR", "Volatilidade", "Beta", "Sharpe Ratio", "Sortino Ratio",
            "Drawdown Máximo", "Worst Case Loss", "Correlação", "Expected Shortfall",
            "Information Ratio", "Treynor Ratio", "Tracking Error"
        ]
        
        self.risk_types = [
            "Risco de Mercado", "Risco de Crédito", "Risco de Liquidez", 
            "Risco Operacional", "Risco de Concentração", "Risco de Contraparte",
            "Risco Cambial", "Risco de Taxa de Juros", "Risco de Eventos"
        ]
        
        self.confidence_levels = [
            "95%", "97.5%", "99%", "99.5%"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa o risco de um portfólio ou posição
        
        Args:
            input_data: Descrição textual ou dicionário com dados de portfólio
            
        Returns:
            Resultado da análise de risco
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "portfolio" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                portfolio_name = input_data.get("portfolio_name", "Não especificado")
                time_horizon = input_data.get("time_horizon", "Não especificado")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                portfolio_name = input_data.get("portfolio_name", "Não especificado")
                time_horizon = input_data.get("time_horizon", "Não especificado")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            portfolio_name = "Não especificado"
            time_horizon = "Não especificado"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise de risco",
                "risk_rating": "indefinido",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise detalhada de risco com base nos seguintes dados:
        
        PORTFÓLIO: {portfolio_name}
        HORIZONTE: {time_horizon}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "risk_rating": (string: "muito baixo", "baixo", "moderado", "alto", "muito alto"),
            "overall_assessment": "avaliação geral do risco",
            "key_risk_metrics": {{
                "volatility": "volatilidade do portfólio",
                "var_95": "Value-at-Risk a 95% de confiança",
                "max_drawdown": "drawdown máximo esperado",
                "sharpe_ratio": "índice de Sharpe"
            }},
            "risk_breakdown": {{
                "market_risk": {{
                    "level": "nível de risco (baixo/moderado/alto)",
                    "sources": [principais fontes de risco de mercado],
                    "metrics": {{métricas relevantes}}
                }},
                "credit_risk": {{
                    "level": "nível de risco (baixo/moderado/alto)",
                    "sources": [principais fontes de risco de crédito],
                    "metrics": {{métricas relevantes}}
                }},
                "liquidity_risk": {{
                    "level": "nível de risco (baixo/moderado/alto)",
                    "sources": [principais fontes de risco de liquidez],
                    "metrics": {{métricas relevantes}}
                }},
                "concentration_risk": {{
                    "level": "nível de risco (baixo/moderado/alto)",
                    "sources": [principais concentrações],
                    "metrics": {{métricas relevantes}}
                }}
            }},
            "stress_tests": [resultados de cenários de estresse],
            "risk_recommendations": [recomendações para mitigação de riscos],
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
                result["portfolio_name"] = portfolio_name
                result["time_horizon"] = time_horizon
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, portfolio_name, time_horizon, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise de risco: {str(e)}",
                "risk_rating": "indefinido",
                "confidence": 0,
                "portfolio_name": portfolio_name,
                "time_horizon": time_horizon,
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
        description = []
        
        # Adicionar dados do portfólio se disponíveis
        if "portfolio" in data:
            portfolio = data["portfolio"]
            
            # Visão geral do portfólio
            description.append("VISÃO GERAL DO PORTFÓLIO:")
            
            # Valor total
            if "total_value" in portfolio:
                description.append(f"Valor Total: {portfolio['total_value']}")
            
            # Alocação por classe de ativo
            if "asset_allocation" in portfolio:
                allocation = portfolio["asset_allocation"]
                description.append("\nALOCAÇÃO POR CLASSE DE ATIVO:")
                
                for asset_class, value in allocation.items():
                    if isinstance(value, dict):
                        allocation_pct = value.get("percentage", "N/A")
                        allocation_value = value.get("value", "N/A")
                        description.append(f"- {asset_class}: {allocation_pct}% ({allocation_value})")
                    else:
                        description.append(f"- {asset_class}: {value}")
            
            # Posições individuais
            if "positions" in portfolio:
                positions = portfolio["positions"]
                description.append("\nPOSIÇÕES INDIVIDUAIS:")
                
                for position in positions:
                    pos_str = []
                    for key, value in position.items():
                        pos_str.append(f"{key}: {value}")
                    description.append("- " + ", ".join(pos_str))
            
            # Exposições 
            if "exposures" in portfolio:
                exposures = portfolio["exposures"]
                description.append("\nEXPOSIÇÕES:")
                
                for exposure_type, exposure_data in exposures.items():
                    description.append(f"{exposure_type.upper()}:")
                    if isinstance(exposure_data, dict):
                        for name, value in exposure_data.items():
                            description.append(f"- {name}: {value}")
                    elif isinstance(exposure_data, list):
                        for item in exposure_data:
                            if isinstance(item, dict):
                                item_str = [f"{k}: {v}" for k, v in item.items()]
                                description.append("- " + ", ".join(item_str))
                            else:
                                description.append(f"- {item}")
                    else:
                        description.append(f"- {exposure_data}")
        
        # Adicionar métricas de risco se disponíveis
        if "risk_metrics" in data:
            metrics = data["risk_metrics"]
            description.append("\nMÉTRICAS DE RISCO:")
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    description.append(f"{metric_name.upper()}:")
                    for key, value in metric_value.items():
                        description.append(f"- {key}: {value}")
                else:
                    description.append(f"- {metric_name}: {metric_value}")
        
        # Adicionar análise de correlação se disponível
        if "correlations" in data:
            correlations = data["correlations"]
            description.append("\nANÁLISE DE CORRELAÇÃO:")
            
            if isinstance(correlations, dict):
                for asset1, corr_data in correlations.items():
                    if isinstance(corr_data, dict):
                        for asset2, corr_value in corr_data.items():
                            description.append(f"- {asset1} vs {asset2}: {corr_value}")
                    else:
                        description.append(f"- {asset1}: {corr_data}")
            elif isinstance(correlations, list):
                for corr_item in correlations:
                    if isinstance(corr_item, dict):
                        asset1 = corr_item.get("asset1", "")
                        asset2 = corr_item.get("asset2", "")
                        value = corr_item.get("value", "")
                        description.append(f"- {asset1} vs {asset2}: {value}")
        
        # Adicionar resultados de teste de estresse se disponíveis
        if "stress_tests" in data:
            stress_tests = data["stress_tests"]
            description.append("\nRESULTADOS DE TESTES DE ESTRESSE:")
            
            for scenario, results in stress_tests.items():
                description.append(f"{scenario.upper()}:")
                if isinstance(results, dict):
                    for key, value in results.items():
                        description.append(f"- {key}: {value}")
                else:
                    description.append(f"- Impacto: {results}")
        
        return "\n".join(description)
    
    def _parse_non_json_response(self, response: str, portfolio_name: str, time_horizon: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações de uma resposta que não está em formato JSON
        
        Args:
            response: Texto da resposta
            portfolio_name: Nome do portfólio
            time_horizon: Horizonte de tempo
            start_time: Tempo de início do processamento
            
        Returns:
            Dicionário com resultados da análise
        """
        result = {
            "risk_rating": "moderado",  # valor padrão
            "overall_assessment": "Análise incompleta",
            "key_risk_metrics": {
                "volatility": "N/A",
                "var_95": "N/A",
                "max_drawdown": "N/A",
                "sharpe_ratio": "N/A"
            },
            "risk_breakdown": {
                "market_risk": {
                    "level": "indefinido",
                    "sources": [],
                    "metrics": {}
                },
                "credit_risk": {
                    "level": "indefinido",
                    "sources": [],
                    "metrics": {}
                },
                "liquidity_risk": {
                    "level": "indefinido",
                    "sources": [],
                    "metrics": {}
                },
                "concentration_risk": {
                    "level": "indefinido",
                    "sources": [],
                    "metrics": {}
                }
            },
            "stress_tests": [],
            "risk_recommendations": [],
            "confidence": 30,
            "portfolio_name": portfolio_name,
            "time_horizon": time_horizon,
            "processing_time": time.time() - start_time
        }
        
        # Extrair classificação de risco
        risk_rating_terms = {
            "muito baixo": ["muito baixo", "very low", "mínimo", "minimal"],
            "baixo": ["baixo", "low", "reduzido", "pequeno"],
            "moderado": ["moderado", "médio", "medium", "moderate", "neutro"],
            "alto": ["alto", "high", "elevado", "significativo"],
            "muito alto": ["muito alto", "very high", "extremo", "severe"]
        }
        
        for rating, keywords in risk_rating_terms.items():
            if any(kw in response.lower() for kw in keywords):
                result["risk_rating"] = rating
                break
        
        # Extrair avaliação geral
        lines = response.split("\n")
        for i, line in enumerate(lines):
            if any(term in line.lower() for term in ["avaliação geral", "overall", "conclusão", "conclusion"]):
                # Pegar as próximas linhas não vazias
                j = i + 1
                assessment_lines = []
                while j < len(lines) and len(assessment_lines) < 3:
                    if lines[j].strip():
                        assessment_lines.append(lines[j].strip())
                    j += 1
                    
                if assessment_lines:
                    result["overall_assessment"] = " ".join(assessment_lines)
                    break
        
        # Extrair métricas de risco
        for metric in ["volatilidade", "volatility", "var", "drawdown", "sharpe"]:
            for line in lines:
                if metric in line.lower() and ":" in line:
                    key = line.split(":", 1)[0].strip().lower()
                    value = line.split(":", 1)[1].strip()
                    
                    if "volatil" in key:
                        result["key_risk_metrics"]["volatility"] = value
                    elif "var" in key or "value at risk" in key:
                        result["key_risk_metrics"]["var_95"] = value
                    elif "drawdown" in key:
                        result["key_risk_metrics"]["max_drawdown"] = value
                    elif "sharpe" in key:
                        result["key_risk_metrics"]["sharpe_ratio"] = value
        
        # Extrair nível de risco para diferentes tipos
        risk_types = ["market", "credit", "liquidity", "concentration"]
        risk_levels = ["baixo", "low", "moderado", "moderate", "medium", "alto", "high"]
        
        for risk_type in risk_types:
            for i, line in enumerate(lines):
                # Verificar se a linha menciona este tipo de risco
                if any(rt in line.lower() for rt in [risk_type, self._translate_risk_type(risk_type)]):
                    # Verificar se esta linha ou as próximas mencionam o nível de risco
                    for j in range(i, min(i + 5, len(lines))):
                        if any(level in lines[j].lower() for level in risk_levels):
                            # Determinar o nível de risco
                            if any(level in lines[j].lower() for level in ["baixo", "low"]):
                                risk_level = "baixo"
                            elif any(level in lines[j].lower() for level in ["alto", "high"]):
                                risk_level = "alto"
                            else:
                                risk_level = "moderado"
                                
                            result["risk_breakdown"][f"{risk_type}_risk"]["level"] = risk_level
                            break
        
        # Extrair recomendações
        recommendations = []
        recommendation_section = False
        
        for line in lines:
            if any(term in line.lower() for term in ["recomendações", "recommendations", "sugestões", "measures"]):
                recommendation_section = True
                continue
                
            if recommendation_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•") or line[0].isdigit()):
                # Remover marcadores
                rec = line.strip()
                if rec[0] in "-*•":
                    rec = rec[1:].strip()
                elif rec[0].isdigit() and rec[1] in ".) ":
                    rec = rec[2:].strip()
                    
                if rec:
                    recommendations.append(rec)
                    
                    # Limitar a 5 recomendações
                    if len(recommendations) >= 5:
                        break
        
        if recommendations:
            result["risk_recommendations"] = recommendations
        
        return result
    
    def _translate_risk_type(self, risk_type: str) -> str:
        """
        Traduz o tipo de risco do inglês para português
        
        Args:
            risk_type: Tipo de risco em inglês
            
        Returns:
            Tipo de risco em português
        """
        translations = {
            "market": "mercado",
            "credit": "crédito",
            "liquidity": "liquidez",
            "concentration": "concentração"
        }
        return translations.get(risk_type, risk_type)
        
    def analyze_correlation(self, assets_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analisa correlações entre diferentes ativos
        
        Args:
            assets_data: Dicionário com listas de retornos para cada ativo
            
        Returns:
            Análise de correlação
        """
        start_time = time.time()
        
        # Verificar se temos dados suficientes
        if not assets_data or len(assets_data) < 2:
            return {
                "error": "São necessários pelo menos dois ativos para análise de correlação",
                "processing_time": time.time() - start_time
            }
        
        # Calcular correlações se tivermos as ferramentas numéricas disponíveis
        try:
            import numpy as np
            
            assets = list(assets_data.keys())
            correlations = {}
            
            # Calcular matriz de correlação
            returns_data = []
            for asset in assets:
                returns_data.append(assets_data[asset])
                
            # Converter para array numpy
            returns_array = np.array(returns_data)
            
            # Calcular matriz de correlação
            corr_matrix = np.corrcoef(returns_array)
            
            # Converter para dicionário
            for i, asset1 in enumerate(assets):
                correlations[asset1] = {}
                for j, asset2 in enumerate(assets):
                    if i != j:  # Não incluir autocorrelação
                        correlations[asset1][asset2] = float(corr_matrix[i, j])
            
            # Encontrar pares com correlação mais alta e mais baixa
            high_corr_pairs = []
            low_corr_pairs = []
            
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i < j:  # Evitar duplicação
                        corr = float(corr_matrix[i, j])
                        pair = {"asset1": asset1, "asset2": asset2, "correlation": corr}
                        
                        if corr >= 0.7:
                            high_corr_pairs.append(pair)
                        elif corr <= 0.3:
                            low_corr_pairs.append(pair)
            
            # Ordenar por correlação
            high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
            low_corr_pairs.sort(key=lambda x: x["correlation"])
            
            # Resumo das correlações
            result = {
                "correlation_matrix": correlations,
                "high_correlation_pairs": high_corr_pairs[:5],  # Top 5
                "low_correlation_pairs": low_corr_pairs[:5],    # Bottom 5
                "average_correlation": float(np.mean([np.mean(row) for row in corr_matrix])),
                "diversification_assessment": "Será calculado pelo modelo",
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            # Fallback para descrição textual se não pudermos calcular
            description = "DADOS DE RETORNOS:\n"
            for asset, returns in assets_data.items():
                description += f"{asset}: {returns}\n"
                
            # Preparar prompt para o modelo
            prompt = f"""
            Analise as correlações entre os seguintes ativos com base em seus retornos:
            
            {description}
            
            Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
            {{
                "correlation_assessment": "avaliação geral das correlações",
                "high_correlation_pairs": [pares com alta correlação],
                "low_correlation_pairs": [pares com baixa correlação],
                "diversification_assessment": "avaliação da diversificação",
                "correlation_insights": [insights sobre correlações]
            }}
            """
            
            response = self.generate_response(prompt)
            
            # Extrair JSON da resposta
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                result["processing_time"] = time.time() - start_time
            else:
                result = {
                    "error": f"Não foi possível calcular correlações: {str(e)}",
                    "assets": list(assets_data.keys()),
                    "processing_time": time.time() - start_time
                }
        
        # Usar o modelo para fornecer insights sobre a diversificação
        assets_list = list(assets_data.keys())
        prompt = f"""
        Com base nos ativos {', '.join(assets_list)}, avalie a diversificação do portfólio.
        
        Dados relevantes:
        - Ativos: {assets_list}
        - Correlação média: {result.get('average_correlation', 'N/A')}
        - Pares com alta correlação: {result.get('high_correlation_pairs', [])}
        - Pares com baixa correlação: {result.get('low_correlation_pairs', [])}
        
        Forneça sua avaliação no seguinte formato JSON (sem explicações adicionais):
        {{
            "diversification_rating": (string: "excelente", "boa", "moderada", "fraca", "pobre"),
            "diversification_assessment": "avaliação detalhada da diversificação",
            "improvements": [sugestões para melhorar a diversificação],
            "risk_reduction_potential": "potencial de redução de risco via diversificação"
        }}
        """
        
        try:
            response = self.generate_response(prompt)
            
            # Extrair JSON da resposta
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                diversification_result = json.loads(json_str)
                
                # Adicionar ao resultado principal
                result.update(diversification_result)
        except:
            # Manter o resultado original se falhar
            pass
            
        return result
    
    def stress_test(self, portfolio: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Realiza testes de estresse em um portfólio
        
        Args:
            portfolio: Dicionário com dados do portfólio
            scenarios: Lista de cenários de estresse
            
        Returns:
            Resultados dos testes de estresse
        """
        start_time = time.time()
        
        # Formatar dados do portfólio
        portfolio_description = self._format_structured_data({"portfolio": portfolio})
        
        # Formatar cenários
        scenarios_description = "CENÁRIOS DE ESTRESSE:\n"
        for i, scenario in enumerate(scenarios):
            scenarios_description += f"Cenário {i+1}: {scenario.get('name', f'Cenário {i+1}')}\n"
            
            if "description" in scenario:
                scenarios_description += f"Descrição: {scenario['description']}\n"
                
            if "parameters" in scenario:
                scenarios_description += "Parâmetros:\n"
                for key, value in scenario["parameters"].items():
                    scenarios_description += f"- {key}: {value}\n"
                    
            scenarios_description += "\n"
            
        # Preparar prompt para o modelo
        prompt = f"""
        Realize testes de estresse no seguinte portfólio com os cenários fornecidos:
        
        {portfolio_description}
        
        {scenarios_description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "stress_test_results": [
                {{
                    "scenario": "nome do cenário",
                    "portfolio_impact": "impacto percentual no portfólio",
                    "var_impact": "impacto no VaR",
                    "key_vulnerabilities": [principais vulnerabilidades],
                    "asset_impacts": [
                        {{
                            "asset": "nome do ativo",
                            "impact": "impacto percentual"
                        }}
                    ]
                }}
            ],
            "overall_assessment": "avaliação geral da resiliência",
            "most_vulnerable_assets": [ativos mais vulneráveis],
            "risk_recommendations": [recomendações para mitigar riscos],
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
                result["portfolio_value"] = portfolio.get("total_value", "N/A")
                result["scenario_count"] = len(scenarios)
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON
                return {
                    "error": "Não foi possível processar os testes de estresse",
                    "scenario_count": len(scenarios),
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            return {
                "error": f"Erro nos testes de estresse: {str(e)}",
                "scenario_count": len(scenarios),
                "processing_time": time.time() - start_time
            }
