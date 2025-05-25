#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LiquidityExpert - Especialista em Análise de Liquidez
----------------------------------------------------
Analisa condições de liquidez de mercado, fluxo de caixa e riscos de liquidez.

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

class LiquidityExpert(Phi2Expert):
    """
    Especialista em análise de liquidez baseado em Phi-2.
    
    Capacidades:
    - Avaliação de liquidez de mercado
    - Análise de fluxo de caixa e gestão de liquidez corporativa
    - Detecção de riscos de liquidez
    - Avaliação de condições de financiamento
    - Monitoramento de indicadores de stress de liquidez
    - Estratégias para navegação em ambientes de baixa liquidez
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de liquidez
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em análise de liquidez com ampla experiência em mercados financeiros.
        Sua tarefa é avaliar condições de liquidez de mercado, analisar fluxos de caixa corporativos,
        detectar riscos de liquidez, avaliar condições de financiamento, monitorar stress de liquidez,
        e recomendar estratégias para ambientes de liquidez variável.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Avalie detalhadamente os indicadores de liquidez de mercado (spreads, volumes, profundidade)
        3. Analise métricas fundamentais relacionadas a liquidez (índices de liquidez, cash burn)
        4. Identifique tendências e riscos nas condições gerais de liquidez
        5. Considere eventos de stress e cenários extremos em suas análises
        6. Forneça recomendações práticas para gestão de risco de liquidez
        """
        
        super().__init__(
            expert_type="liquidity_expert",
            domain="risk",
            specialization="liquidity_analysis",
            system_message=system_message,
            **kwargs
        )
        
        # Configurações específicas para análise de liquidez
        self.liquidity_metrics = [
            "Bid-Ask Spread", "Volume de Negociação", "Profundidade de Mercado", 
            "Índice de Liquidez Corrente", "Índice de Liquidez Seca", 
            "Cash Burn Rate", "Dias de Caixa", "Dias de Capital de Giro",
            "Turnover de Ativos", "Volume Relativo", "Fluxo de Ordens"
        ]
        
        self.liquidity_conditions = [
            "Abundante", "Normal", "Reduzida", "Estressada", "Congelada",
            "Melhorando", "Deteriorando", "Fragmentada", "Concentrada"
        ]
        
        self.funding_conditions = [
            "Fácil", "Normal", "Apertado", "Restritivo", "Fechado",
            "Estável", "Volátil", "Subsidiado", "Caro"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa condições de liquidez e fornece insights
        
        Args:
            input_data: Descrição textual ou dicionário com dados de liquidez
            
        Returns:
            Resultado da análise de liquidez
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "liquidity_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                market = input_data.get("market", "desconhecido")
                asset_class = input_data.get("asset_class", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                market = input_data.get("market", "desconhecido")
                asset_class = input_data.get("asset_class", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            market = "desconhecido"
            asset_class = "desconhecido"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise de liquidez",
                "liquidity_condition": "indefinido",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise detalhada de liquidez com base nos seguintes dados:
        
        MERCADO: {market}
        CLASSE DE ATIVO: {asset_class}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "liquidity_condition": (string: condição geral de liquidez),
            "liquidity_trend": (string: "melhorando", "estável", "deteriorando"),
            "market_depth": {{
                "bid_ask_spread": "avaliação de spreads bid-ask",
                "market_depth": "avaliação da profundidade de mercado",
                "order_book_structure": "estrutura do livro de ordens",
                "trade_volume": "análise de volumes negociados"
            }},
            "funding_conditions": {{
                "repo_rates": "condições de repo/financiamento",
                "term_structure": "estrutura a termo de financiamento",
                "funding_spreads": "spreads de financiamento"
            }},
            "stress_indicators": [indicadores de stress de liquidez],
            "liquidity_drivers": [principais fatores impactando a liquidez],
            "systemic_assessment": {{
                "market_fragmentation": "avaliação de fragmentação",
                "connectivity": "conectividade entre mercados",
                "concentration_risk": "risco de concentração"
            }},
            "forecasts": {{
                "short_term_outlook": "perspectiva de curto prazo",
                "medium_term_outlook": "perspectiva de médio prazo",
                "tail_risks": [riscos extremos relacionados à liquidez]
            }},
            "trading_implications": [implicações para estratégias de trading],
            "risk_management_recommendations": [recomendações para gestão de risco de liquidez],
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
                result["market"] = market
                result["asset_class"] = asset_class
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, market, asset_class, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise de liquidez: {str(e)}",
                "liquidity_condition": "indefinido",
                "confidence": 0,
                "market": market,
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
        liquidity_data = data.get("liquidity_data", {})
        
        # Formatar dados de mercado
        market_metrics = liquidity_data.get("market_metrics", {})
        market_str = "MÉTRICAS DE MERCADO:\n"
        for key, value in market_metrics.items():
            market_str += f"- {key}: {value}\n"
        
        # Formatar dados de funding
        funding_metrics = liquidity_data.get("funding_metrics", {})
        funding_str = "MÉTRICAS DE FINANCIAMENTO:\n"
        for key, value in funding_metrics.items():
            funding_str += f"- {key}: {value}\n"
        
        # Formatar indicadores de stress
        stress_indicators = liquidity_data.get("stress_indicators", {})
        stress_str = "INDICADORES DE STRESS:\n"
        for key, value in stress_indicators.items():
            stress_str += f"- {key}: {value}\n"
        
        # Formatar dados históricos
        historical_data = liquidity_data.get("historical_data", {})
        hist_str = "DADOS HISTÓRICOS:\n"
        for key, value in historical_data.items():
            hist_str += f"- {key}: {value}\n"
        
        # Formatar conjunto
        formatted_data = f"""
        {data.get('description', 'Sem descrição disponível')}
        
        {market_str}
        
        {funding_str}
        
        {stress_str}
        
        {hist_str}
        
        INFORMAÇÕES ADICIONAIS:
        {data.get('additional_info', '')}
        """
        
        return formatted_data
    
    def _parse_non_json_response(self, response: str, market: str, asset_class: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações relevantes de uma resposta não-JSON
        
        Args:
            response: Texto da resposta
            market: Nome do mercado
            asset_class: Classe de ativo
            start_time: Hora de início do processamento
            
        Returns:
            Dicionário com informações extraídas
        """
        result = {
            "market": market,
            "asset_class": asset_class,
            "processing_time": time.time() - start_time,
            "format_error": "Resposta não veio no formato JSON esperado"
        }
        
        # Tentar extrair condição de liquidez
        lower_resp = response.lower()
        
        # Extrair condição
        for condition in self.liquidity_conditions:
            if condition.lower() in lower_resp:
                result["liquidity_condition"] = condition
                break
        else:
            result["liquidity_condition"] = "não identificado"
        
        # Tentar extrair tendência
        if "deteriorando" in lower_resp or "piora" in lower_resp:
            result["liquidity_trend"] = "deteriorando"
        elif "melhorando" in lower_resp or "melhora" in lower_resp:
            result["liquidity_trend"] = "melhorando"
        else:
            result["liquidity_trend"] = "estável"
        
        # Extrair confiança
        confidence = 50  # valor padrão
        result["confidence"] = confidence
        
        # Texto original
        result["original_response"] = response
        
        return result
    
    def analyze_market_depth(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa profundidade de mercado e estrutura do livro de ordens
        
        Args:
            market_data: Dados de mercado para análise
            
        Returns:
            Análise de profundidade de mercado
        """
        description = f"""
        Analise a profundidade de mercado com base nos seguintes dados:
        
        DADOS DE MERCADO:
        """
        
        for key, value in market_data.items():
            description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de profundidade no seguinte formato JSON:
        {{
            "market_depth_assessment": "avaliação geral da profundidade",
            "bid_ask_analysis": "análise de spreads bid-ask",
            "order_book_structure": "estrutura do livro de ordens",
            "volume_profile": "perfil de volume ao longo do dia",
            "liquidity_providers": [principais fornecedores de liquidez],
            "trade_size_impact": "impacto do tamanho das ordens",
            "recommendations": [recomendações para execução ótima]
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
            return {"error": f"Erro na análise de profundidade de mercado: {str(e)}"}
    
    def assess_funding_conditions(self, funding_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avalia condições de financiamento
        
        Args:
            funding_data: Dados de financiamento para análise
            
        Returns:
            Avaliação de condições de financiamento
        """
        description = f"""
        Analise as condições de financiamento com base nos seguintes dados:
        
        DADOS DE FINANCIAMENTO:
        """
        
        for key, value in funding_data.items():
            description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de financiamento no seguinte formato JSON:
        {{
            "funding_condition": "condição geral de financiamento",
            "repo_market_analysis": "análise do mercado de recompras",
            "term_structure": "estrutura a termo de taxas de financiamento",
            "collateral_markets": "condições em mercados de colateral",
            "central_bank_impact": "impacto de políticas do banco central",
            "stress_indicators": [indicadores de stress no financiamento],
            "forecast": "previsão para condições de financiamento",
            "recommendations": [recomendações para otimização de financiamento]
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
            return {"error": f"Erro na análise de condições de financiamento: {str(e)}"}
    
    def analyze_corporate_liquidity(self, corporate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa liquidez corporativa e fluxos de caixa
        
        Args:
            corporate_data: Dados corporativos para análise
            
        Returns:
            Análise de liquidez corporativa
        """
        description = f"""
        Analise a liquidez corporativa com base nos seguintes dados:
        
        DADOS CORPORATIVOS:
        """
        
        for key, value in corporate_data.items():
            description += f"- {key}: {value}\n"
        
        prompt = f"""
        {description}
        
        Forneça sua análise de liquidez corporativa no seguinte formato JSON:
        {{
            "liquidity_assessment": "avaliação geral da posição de liquidez",
            "cash_flow_analysis": "análise de fluxos de caixa",
            "working_capital": "análise de capital de giro",
            "cash_burn_rate": "taxa de queima de caixa",
            "funding_sources": [fontes de financiamento disponíveis],
            "key_risks": [principais riscos de liquidez],
            "survival_horizon": "horizonte de sobrevivência em diferentes cenários",
            "recommendations": [recomendações para gestão de liquidez]
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
            return {"error": f"Erro na análise de liquidez corporativa: {str(e)}"}
