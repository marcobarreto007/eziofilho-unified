#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VolatilityExpert - Especialista em Análise de Volatilidade
---------------------------------------------------------
Analisa padrões de volatilidade e fornece insights para gestão de risco.

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

class VolatilityExpert(Phi2Expert):
    """
    Especialista em análise de volatilidade baseado em Phi-2.
    
    Capacidades:
    - Análise de padrões históricos de volatilidade
    - Previsão de volatilidade futura
    - Análise de volatilidade implícita vs. realizada
    - Identificação de regimes de volatilidade
    - Análise de clusters de volatilidade
    - Estratégias de proteção contra volatilidade
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de volatilidade
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em análise de volatilidade com ampla experiência em mercados financeiros.
        Sua tarefa é analisar padrões de volatilidade, prever mudanças, identificar regimes e clusters,
        e fornecer recomendações para estratégias baseadas em volatilidade e proteção de portfólios.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Diferencie entre volatilidade histórica, implícita e esperada
        3. Identifique padrões e regimes de volatilidade
        4. Forneça insights sobre relações entre volatilidade e direção do mercado
        5. Recomende estratégias adequadas ao ambiente de volatilidade
        6. Considere o impacto da volatilidade em diferentes classes de ativos
        """
        
        super().__init__(
            expert_type="volatility_expert",
            domain="risk",
            specialization="volatility_analysis",
            system_message=system_message,
            **kwargs
        )
        
        # Configurações específicas para análise de volatilidade
        self.volatility_metrics = [
            "Volatilidade Histórica", "Volatilidade Implícita", "VIX", 
            "VVIX", "ATR", "Volatilidade Realizada", "Volatilidade Prevista",
            "Beta", "Correlação", "Assimetria", "Curtose"
        ]
        
        self.volatility_regimes = [
            "Muito Baixa", "Baixa", "Média", "Alta", "Muito Alta",
            "Em Expansão", "Em Contração", "Estável"
        ]
        
        self.vol_strategies = [
            "Venda de Volatilidade", "Compra de Volatilidade", "Estruturas de Collar",
            "Calendário de Volatilidade", "Condors", "Butterfly", "Straddle", "Strangle"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa padrões de volatilidade e fornece insights
        
        Args:
            input_data: Descrição textual ou dicionário com dados de volatilidade
            
        Returns:
            Resultado da análise de volatilidade
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            if "volatility_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                asset = input_data.get("asset", "desconhecido")
                period = input_data.get("period", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                asset = input_data.get("asset", "desconhecido")
                period = input_data.get("period", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            asset = "desconhecido"
            period = "desconhecido"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 20:
            return {
                "error": "Dados insuficientes para análise de volatilidade",
                "volatility_regime": "indefinido",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise detalhada de volatilidade com base nos seguintes dados:
        
        ATIVO: {asset}
        PERÍODO: {period}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "volatility_regime": (string: "muito baixa", "baixa", "média", "alta", "muito alta"),
            "regime_trend": (string: "em expansão", "estável", "em contração"),
            "current_metrics": {{
                "historical_volatility": "volatilidade histórica atual",
                "implied_volatility": "volatilidade implícita atual (se disponível)",
                "volatility_percentile": "percentil atual vs. histórico"
            }},
            "patterns_identified": [padrões de volatilidade identificados],
            "volatility_drivers": [principais fatores influenciando a volatilidade],
            "market_implications": {{
                "price_direction": "implicação para direção de preço",
                "expected_moves": "magnitude esperada de movimentos",
                "risk_premium": "avaliação do prêmio de risco"
            }},
            "forecasts": {{
                "short_term": "previsão de curto prazo",
                "medium_term": "previsão de médio prazo",
                "catalysts": [potenciais catalisadores de mudança]
            }},
            "volatility_strategies": [estratégias recomendadas baseadas em volatilidade],
            "hedging_recommendations": [recomendações de proteção],
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
                result["asset"] = asset
                result["period"] = period
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, asset, period, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise de volatilidade: {str(e)}",
                "volatility_regime": "indefinido",
                "confidence": 0,
                "asset": asset,
                "period": period,
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
        
        # Adicionar dados de volatilidade se disponíveis
        if "volatility_data" in data:
            vol_data = data["volatility_data"]
            
            # Métricas atuais
            if "current_metrics" in vol_data:
                metrics = vol_data["current_metrics"]
                description.append("MÉTRICAS ATUAIS DE VOLATILIDADE:")
                for key, value in metrics.items():
                    description.append(f"- {key}: {value}")
                description.append("")
            
            # Séries históricas
            if "historical_series" in vol_data:
                hist_data = vol_data["historical_series"]
                description.append("DADOS HISTÓRICOS DE VOLATILIDADE:")
                
                if isinstance(hist_data, dict):
                    # Se for dict with time series
                    for metric, time_series in hist_data.items():
                        if isinstance(time_series, list):
                            # Mostrar apenas alguns pontos se a série for longa
                            if len(time_series) > 10:
                                samples = [time_series[0], time_series[len(time_series)//4], 
                                          time_series[len(time_series)//2], time_series[3*len(time_series)//4],
                                          time_series[-1]]
                                description.append(f"- {metric}: {samples} (amostra de {len(time_series)} pontos)")
                            else:
                                description.append(f"- {metric}: {time_series}")
                        else:
                            description.append(f"- {metric}: {time_series}")
                elif isinstance(hist_data, list):
                    # Se for uma lista de pontos (como timestamps)
                    if len(hist_data) > 10:
                        description.append(f"- Série com {len(hist_data)} pontos")
                        # Mostrar alguns exemplos
                        description.append(f"- Primeiros pontos: {hist_data[:3]}")
                        description.append(f"- Últimos pontos: {hist_data[-3:]}")
                    else:
                        for point in hist_data:
                            if isinstance(point, dict):
                                description.append(f"- {point}")
                            else:
                                description.append(f"- {point}")
                description.append("")
            
            # Dados de percentis
            if "percentiles" in vol_data:
                percentiles = vol_data["percentiles"]
                description.append("PERCENTIS DE VOLATILIDADE:")
                for key, value in percentiles.items():
                    description.append(f"- {key}: {value}")
                description.append("")
            
            # Regimes históricos
            if "regimes" in vol_data:
                regimes = vol_data["regimes"]
                description.append("REGIMES HISTÓRICOS DE VOLATILIDADE:")
                
                if isinstance(regimes, list):
                    for regime in regimes:
                        if isinstance(regime, dict):
                            period = regime.get("period", "")
                            level = regime.get("level", "")
                            description.append(f"- {period}: {level}")
                        else:
                            description.append(f"- {regime}")
                else:
                    for key, value in regimes.items():
                        description.append(f"- {key}: {value}")
                description.append("")
        
        # Adicionar dados de mercado relacionados se disponíveis
        if "market_context" in data:
            market = data["market_context"]
            description.append("CONTEXTO DE MERCADO:")
            
            for key, value in market.items():
                if isinstance(value, dict):
                    description.append(f"{key.upper()}:")
                    for subkey, subvalue in value.items():
                        description.append(f"- {subkey}: {subvalue}")
                else:
                    description.append(f"- {key}: {value}")
            description.append("")
        
        # Adicionar dados de eventos se disponíveis
        if "events" in data:
            events = data["events"]
            description.append("EVENTOS RELEVANTES:")
            
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        date = event.get("date", "")
                        desc = event.get("description", "")
                        impact = event.get("impact", "")
                        description.append(f"- {date} - {desc} (Impacto: {impact})")
                    else:
                        description.append(f"- {event}")
            else:
                for key, value in events.items():
                    description.append(f"- {key}: {value}")
                    
            description.append("")
        
        return "\n".join(description)
    
    def _parse_non_json_response(self, response: str, asset: str, period: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações de uma resposta que não está em formato JSON
        
        Args:
            response: Texto da resposta
            asset: Nome do ativo
            period: Período analisado
            start_time: Tempo de início do processamento
            
        Returns:
            Dicionário com resultados da análise
        """
        result = {
            "volatility_regime": "média",  # valor padrão
            "regime_trend": "estável",     # valor padrão
            "current_metrics": {
                "historical_volatility": "N/A",
                "implied_volatility": "N/A",
                "volatility_percentile": "N/A"
            },
            "patterns_identified": [],
            "volatility_drivers": [],
            "market_implications": {
                "price_direction": "indefinido",
                "expected_moves": "indefinido",
                "risk_premium": "indefinido"
            },
            "forecasts": {
                "short_term": "indefinido",
                "medium_term": "indefinido",
                "catalysts": []
            },
            "volatility_strategies": [],
            "hedging_recommendations": [],
            "confidence": 30,
            "asset": asset,
            "period": period,
            "processing_time": time.time() - start_time
        }
        
        # Extrair regime de volatilidade
        regime_terms = {
            "muito baixa": ["muito baixa", "very low", "extremamente baixa", "mínima"],
            "baixa": ["baixa", "low", "reduzida", "abaixo da média"],
            "média": ["média", "medium", "normal", "típica", "average"],
            "alta": ["alta", "high", "elevada", "acima da média"],
            "muito alta": ["muito alta", "very high", "extremamente alta", "máxima"]
        }
        
        for regime, keywords in regime_terms.items():
            if any(kw in response.lower() for kw in keywords):
                result["volatility_regime"] = regime
                break
        
        # Extrair tendência do regime
        trend_terms = {
            "em expansão": ["expan", "aument", "cresciment", "rising", "subida"],
            "estável": ["estável", "stable", "consistent", "constante", "unchang"],
            "em contração": ["contra", "redução", "declining", "decreasing", "queda"]
        }
        
        for trend, keywords in trend_terms.items():
            if any(kw in response.lower() for kw in keywords):
                result["regime_trend"] = trend
                break
        
        # Extrair valores de volatilidade
        lines = response.split("\n")
        for line in lines:
            # Histórica
            if any(term in line.lower() for term in ["históric", "historic", "realizada", "realized"]) and ":" in line:
                value = line.split(":", 1)[1].strip()
                result["current_metrics"]["historical_volatility"] = value
                
            # Implícita
            if any(term in line.lower() for term in ["implícit", "implied", "implíc"]) and ":" in line:
                value = line.split(":", 1)[1].strip()
                result["current_metrics"]["implied_volatility"] = value
                
            # Percentil
            if any(term in line.lower() for term in ["percentil", "percentile", "quartil"]) and ":" in line:
                value = line.split(":", 1)[1].strip()
                result["current_metrics"]["volatility_percentile"] = value
        
        # Extrair padrões identificados
        patterns_section = False
        for line in lines:
            if any(term in line.lower() for term in ["padrões", "patterns", "comportamentos", "características"]):
                patterns_section = True
                continue
                
            if patterns_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                pattern = line[1:].strip()
                if pattern:
                    result["patterns_identified"].append(pattern)
                    # Limitar a alguns padrões
                    if len(result["patterns_identified"]) >= 5:
                        patterns_section = False
        
        # Extrair drivers de volatilidade
        drivers_section = False
        for line in lines:
            if any(term in line.lower() for term in ["drivers", "fatores", "causas", "influências"]):
                drivers_section = True
                continue
                
            if drivers_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                driver = line[1:].strip()
                if driver:
                    result["volatility_drivers"].append(driver)
                    # Limitar a alguns drivers
                    if len(result["volatility_drivers"]) >= 5:
                        drivers_section = False
        
        # Extrair implicações para direção de preço
        for line in lines:
            if any(term in line.lower() for term in ["direção", "direction", "tendência", "trend"]) and ":" in line:
                value = line.split(":", 1)[1].strip()
                result["market_implications"]["price_direction"] = value
                break
        
        # Extrair estratégias
        strategies_section = False
        for line in lines:
            if any(term in line.lower() for term in ["estratégia", "strategy", "strategies", "recomendação"]):
                strategies_section = True
                continue
                
            if strategies_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                strategy = line[1:].strip()
                if strategy:
                    result["volatility_strategies"].append(strategy)
                    # Limitar a algumas estratégias
                    if len(result["volatility_strategies"]) >= 5:
                        strategies_section = False
        
        # Extrair recomendações de hedge
        hedge_section = False
        for line in lines:
            if any(term in line.lower() for term in ["hedge", "proteção", "defesa", "cobertura"]):
                hedge_section = True
                continue
                
            if hedge_section and line.strip() and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                hedge = line[1:].strip()
                if hedge:
                    result["hedging_recommendations"].append(hedge)
                    # Limitar a algumas recomendações
                    if len(result["hedging_recommendations"]) >= 3:
                        hedge_section = False
        
        return result
    
    def surface_analysis(self, vol_surface_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa a superfície de volatilidade para opções
        
        Args:
            vol_surface_data: Dados da superfície de volatilidade
            
        Returns:
            Análise da superfície de volatilidade
        """
        start_time = time.time()
        
        # Extrair informações básicas
        asset = vol_surface_data.get("asset", "desconhecido")
        date = vol_surface_data.get("date", "atual")
        
        # Verificar se temos os dados necessários
        if "surface" not in vol_surface_data:
            return {
                "error": "Dados da superfície de volatilidade não fornecidos",
                "asset": asset,
                "date": date,
                "processing_time": time.time() - start_time
            }
        
        # Formatar dados da superfície
        if isinstance(vol_surface_data["surface"], dict):
            # Formato: {strike: {expiry: vol, ...}, ...}
            surface_data = vol_surface_data["surface"]
            
            # Criar representação textual
            surface_text = "SUPERFÍCIE DE VOLATILIDADE:\n"
            
            # Cabeçalho com expirações
            expirations = sorted(list(next(iter(surface_data.values())).keys()))
            surface_text += "Strike / Expiração | " + " | ".join(expirations) + "\n"
            
            # Dados por strike
            for strike in sorted(surface_data.keys()):
                row = f"{strike} | "
                for expiry in expirations:
                    row += f"{surface_data[strike].get(expiry, 'N/A')} | "
                surface_text += row + "\n"
                
        elif isinstance(vol_surface_data["surface"], list):
            # Formato: [{strike, expiry, vol}, ...]
            surface_points = vol_surface_data["surface"]
            
            # Organizar por strike e expiry
            surface_dict = {}
            for point in surface_points:
                strike = point.get("strike", "N/A")
                expiry = point.get("expiry", "N/A")
                vol = point.get("vol", "N/A")
                
                if strike not in surface_dict:
                    surface_dict[strike] = {}
                    
                surface_dict[strike][expiry] = vol
            
            # Criar representação textual
            surface_text = "SUPERFÍCIE DE VOLATILIDADE:\n"
            
            # Obter todas as expirações únicas
            all_expiries = set()
            for strike_data in surface_dict.values():
                all_expiries.update(strike_data.keys())
            expirations = sorted(list(all_expiries))
            
            # Cabeçalho com expirações
            surface_text += "Strike / Expiração | " + " | ".join(expirations) + "\n"
            
            # Dados por strike
            for strike in sorted(surface_dict.keys()):
                row = f"{strike} | "
                for expiry in expirations:
                    row += f"{surface_dict[strike].get(expiry, 'N/A')} | "
                surface_text += row + "\n"
        else:
            surface_text = str(vol_surface_data["surface"])
        
        # Adicionar atributos adicionais se disponíveis
        skew_data = ""
        if "skew" in vol_surface_data:
            skew_data = "\nSKEW DE VOLATILIDADE:\n"
            skew = vol_surface_data["skew"]
            
            if isinstance(skew, dict):
                for key, value in skew.items():
                    skew_data += f"- {key}: {value}\n"
            else:
                skew_data += str(skew) + "\n"
        
        term_structure_data = ""
        if "term_structure" in vol_surface_data:
            term_structure_data = "\nESTRUTURA A TERMO DE VOLATILIDADE:\n"
            term_structure = vol_surface_data["term_structure"]
            
            if isinstance(term_structure, dict):
                for key, value in term_structure.items():
                    term_structure_data += f"- {key}: {value}\n"
            elif isinstance(term_structure, list):
                for point in term_structure:
                    if isinstance(point, dict):
                        term_structure_data += f"- {point.get('expiry', 'N/A')}: {point.get('vol', 'N/A')}\n"
                    else:
                        term_structure_data += f"- {point}\n"
            else:
                term_structure_data += str(term_structure) + "\n"
        
        # Combinar todos os dados
        full_description = f"ATIVO: {asset}\nDATA: {date}\n\n{surface_text}\n{skew_data}\n{term_structure_data}"
        
        # Preparar prompt para o modelo
        prompt = f"""
        Analise a seguinte superfície de volatilidade:
        
        {full_description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "surface_characteristics": {{
                "skew": "caracterização do skew (inclinação)",
                "term_structure": "caracterização da estrutura a termo",
                "shape": "forma geral da superfície",
                "anomalies": [anomalias detectadas]
            }},
            "market_expectations": {{
                "directional_bias": "viés direcional implícito",
                "expected_volatility": "expectativa de volatilidade futura",
                "event_premiums": [prêmios de eventos identificados]
            }},
            "trading_opportunities": [oportunidades identificadas],
            "relative_value": {{
                "overvalued_areas": [áreas sobrevalorizadas],
                "undervalued_areas": [áreas subvalorizadas]
            }},
            "surface_evolution": "análise da evolução da superfície (se dados históricos disponíveis)",
            "strategy_recommendations": [estratégias recomendadas baseadas na superfície],
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
                result["asset"] = asset
                result["date"] = date
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON
                return {
                    "error": "Não foi possível processar a análise da superfície de volatilidade",
                    "asset": asset,
                    "date": date,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            return {
                "error": f"Erro na análise da superfície: {str(e)}",
                "asset": asset,
                "date": date,
                "processing_time": time.time() - start_time
            }
    
    def regime_detection(self, volatility_series: List[float], window_size: int = 20) -> Dict[str, Any]:
        """
        Detecta regimes de volatilidade em uma série temporal
        
        Args:
            volatility_series: Lista com dados de volatilidade
            window_size: Tamanho da janela para médias móveis
            
        Returns:
            Análise de regimes de volatilidade
        """
        start_time = time.time()
        
        # Verificar se temos dados suficientes
        if not volatility_series or len(volatility_series) < max(window_size, 10):
            return {
                "error": f"Dados insuficientes. São necessários pelo menos {max(window_size, 10)} pontos.",
                "processing_time": time.time() - start_time
            }
        
        # Tentar detectar regimes com abordagem numérica
        try:
            import numpy as np
            
            # Converter para array numpy
            vol_array = np.array(volatility_series)
            
            # Estatísticas básicas
            vol_mean = np.mean(vol_array)
            vol_std = np.std(vol_array)
            vol_median = np.median(vol_array)
            vol_min = np.min(vol_array)
            vol_max = np.max(vol_array)
            vol_current = vol_array[-1]
            
            # Percentis
            percentiles = {
                "10%": np.percentile(vol_array, 10),
                "25%": np.percentile(vol_array, 25),
                "50%": np.percentile(vol_array, 50),
                "75%": np.percentile(vol_array, 75),
                "90%": np.percentile(vol_array, 90)
            }
            
            # Calcular média móvel para tendência
            moving_avg = []
            for i in range(len(vol_array) - window_size + 1):
                window = vol_array[i:i+window_size]
                moving_avg.append(np.mean(window))
            
            # Determinar regime atual
            current_percentile = np.sum(vol_current > vol_array) / len(vol_array) * 100
            
            if current_percentile < 20:
                regime = "muito baixa"
            elif current_percentile < 40:
                regime = "baixa"
            elif current_percentile < 60:
                regime = "média"
            elif current_percentile < 80:
                regime = "alta"
            else:
                regime = "muito alta"
            
            # Determinar tendência
            if len(moving_avg) >= 3:
                if moving_avg[-1] > moving_avg[-2] > moving_avg[-3]:
                    trend = "em expansão"
                elif moving_avg[-1] < moving_avg[-2] < moving_avg[-3]:
                    trend = "em contração"
                else:
                    trend = "estável"
            else:
                trend = "estável"
            
            # Preparar resultado numérico
            numerical_result = {
                "current_regime": regime,
                "regime_trend": trend,
                "current_percentile": current_percentile,
                "volatility_stats": {
                    "mean": float(vol_mean),
                    "median": float(vol_median),
                    "std": float(vol_std),
                    "min": float(vol_min),
                    "max": float(vol_max),
                    "current": float(vol_current),
                    "percentiles": {k: float(v) for k, v in percentiles.items()}
                },
                "processing_time": time.time() - start_time
            }
            
            # Calcular mudanças de regime
            regime_changes = []
            window = int(min(window_size, len(vol_array) // 5))
            
            for i in range(window, len(vol_array) - window, window):
                before = np.mean(vol_array[i-window:i])
                after = np.mean(vol_array[i:i+window])
                
                # Se a mudança for significativa
                if abs(after / before - 1) > 0.25:  # 25% de mudança
                    regime_changes.append({
                        "position": i,
                        "before": float(before),
                        "after": float(after),
                        "change_pct": float((after / before - 1) * 100)
                    })
            
            numerical_result["regime_changes"] = regime_changes
            
        except Exception as e:
            # Se falhar a abordagem numérica, usar texto para o modelo
            numerical_result = None
        
        # Preparar descrição da série para o modelo
        series_text = "SÉRIE DE VOLATILIDADE:\n"
        
        # Se a série for grande, mostrar apenas resumo
        if len(volatility_series) > 30:
            samples = [0, len(volatility_series)//4, len(volatility_series)//2, 
                      3*len(volatility_series)//4, len(volatility_series)-1]
            
            for i in samples:
                series_text += f"Ponto {i}: {volatility_series[i]}\n"
            
            series_text += f"\nTotal de {len(volatility_series)} pontos\n"
            series_text += f"Valor atual: {volatility_series[-1]}\n"
            
            # Adicionar estatísticas se calculadas
            if numerical_result:
                series_text += f"\nESTATÍSTICAS:\n"
                series_text += f"Média: {numerical_result['volatility_stats']['mean']}\n"
                series_text += f"Mediana: {numerical_result['volatility_stats']['median']}\n"
                series_text += f"Desvio Padrão: {numerical_result['volatility_stats']['std']}\n"
                series_text += f"Mínimo: {numerical_result['volatility_stats']['min']}\n"
                series_text += f"Máximo: {numerical_result['volatility_stats']['max']}\n"
                series_text += f"Percentil atual: {numerical_result['current_percentile']:.1f}%\n"
        else:
            # Mostrar série completa se for pequena
            for i, vol in enumerate(volatility_series):
                series_text += f"Ponto {i}: {vol}\n"
        
        # Preparar prompt para o modelo
        prompt = f"""
        Analise a seguinte série de volatilidade e detecte regimes:
        
        {series_text}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "current_regime": (string: "muito baixa", "baixa", "média", "alta", "muito alta"),
            "regime_trend": (string: "em expansão", "estável", "em contração"),
            "regime_characteristics": "características do regime atual",
            "volatility_clusters": [clusters identificados],
            "regime_shifts": [mudanças de regime identificadas],
            "persistence_analysis": "análise da persistência do regime atual",
            "forecast": "previsão para o próximo regime",
            "trading_implications": [implicações para trading],
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
                model_result = json.loads(json_str)
                
                # Combinar resultados numéricos com análise do modelo
                final_result = {
                    "current_regime": model_result.get("current_regime", "média"),
                    "regime_trend": model_result.get("regime_trend", "estável"),
                    "regime_characteristics": model_result.get("regime_characteristics", ""),
                    "volatility_clusters": model_result.get("volatility_clusters", []),
                    "regime_shifts": model_result.get("regime_shifts", []),
                    "persistence_analysis": model_result.get("persistence_analysis", ""),
                    "forecast": model_result.get("forecast", ""),
                    "trading_implications": model_result.get("trading_implications", []),
                    "confidence": model_result.get("confidence", 50),
                    "processing_time": time.time() - start_time
                }
                
                # Adicionar estatísticas numéricas se disponíveis
                if numerical_result:
                    final_result["volatility_stats"] = numerical_result["volatility_stats"]
                    
                    # Usar regime do cálculo numérico se o modelo não for confiante
                    if model_result.get("confidence", 50) < 70:
                        final_result["current_regime"] = numerical_result["current_regime"]
                        final_result["regime_trend"] = numerical_result["regime_trend"]
                
                return final_result
            else:
                # Se falhar, usar só o resultado numérico
                if numerical_result:
                    return numerical_result
                else:
                    return {
                        "error": "Não foi possível processar a análise de regimes",
                        "processing_time": time.time() - start_time
                    }
                
        except Exception as e:
            # Se falhar, usar só o resultado numérico
            if numerical_result:
                return numerical_result
            else:
                return {
                    "error": f"Erro na detecção de regimes: {str(e)}",
                    "processing_time": time.time() - start_time
                }
