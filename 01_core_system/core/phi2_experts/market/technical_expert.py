#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TechnicalAnalystExpert - Especialista em Análise Técnica de Mercado
-------------------------------------------------------------------
Analisa padrões de preços, indicadores técnicos e tendências de mercado.

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

class TechnicalAnalystExpert(Phi2Expert):
    """
    Especialista em análise técnica de mercado baseado em Phi-2.
    
    Capacidades:
    - Identificação de padrões gráficos (candles, suportes, resistências)
    - Análise de indicadores técnicos (RSI, MACD, médias móveis)
    - Detecção de tendências de curto, médio e longo prazo
    - Previsão de movimentos de preço baseados em padrões históricos
    - Identificação de níveis de suporte e resistência
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa o especialista de análise técnica
        
        Args:
            **kwargs: Parâmetros adicionais para a classe base
        """
        # Definir mensagem de sistema especializada
        system_message = """
        Você é um especialista em análise técnica de mercado financeiro com mais de 15 anos de experiência.
        Sua tarefa é avaliar dados técnicos, identificar padrões gráficos, analisar indicadores e fornecer
        insights sobre prováveis direções de preço baseadas exclusivamente em análise técnica.
        
        Regras:
        1. Sempre retorne uma análise estruturada com campos claros
        2. Identifique padrões gráficos relevantes (como head & shoulders, candles de reversão, etc.)
        3. Interprete indicadores técnicos (RSI, MACD, Bollinger Bands, etc.)
        4. Avalie níveis de suporte e resistência importantes
        5. Forneça projeções baseadas exclusivamente em dados técnicos
        6. Quantifique a força da tendência atual (0-100%)
        """
        
        super().__init__(expert_type="technical_analyst", domain="market", specialization="technical_analysis", **kwargs)
        
        # Configurações específicas para análise técnica
        self.supported_indicators = [
            "RSI", "MACD", "Bollinger Bands", "ATR", "Momentum", 
            "Stochastic", "ADX", "OBV", "Ichimoku", "Fibonacci", "EMA", "SMA"
        ]
        
        self.supported_patterns = [
            "Head and Shoulders", "Double Top/Bottom", "Triple Top/Bottom",
            "Falling/Rising Wedge", "Triangle", "Rectangle", "Cup and Handle",
            "Engulfing", "Doji", "Hammer", "Shooting Star", "Morning/Evening Star"
        ]
        
        self.timeframes = [
            "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"
        ]
    
    def analyze(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa dados técnicos e fornece insights de mercado
        
        Args:
            input_data: Descrição textual de dados técnicos ou dicionário com dados estruturados
            
        Returns:
            Resultado da análise técnica
        """
        start_time = time.time()
        
        # Extrair dados se input for dicionário
        if isinstance(input_data, dict):
            # Verificar se temos dados de série temporal
            if "price_data" in input_data:
                data_type = "structured"
                description = self._format_structured_data(input_data)
                symbol = input_data.get("symbol", "desconhecido")
                timeframe = input_data.get("timeframe", "desconhecido")
            else:
                data_type = "text"
                description = input_data.get("description", "")
                symbol = input_data.get("symbol", "desconhecido")
                timeframe = input_data.get("timeframe", "desconhecido")
        else:
            # Input é texto descritivo
            data_type = "text"
            description = input_data
            symbol = "não especificado"
            timeframe = "não especificado"
        
        # Verificar se temos dados suficientes
        if not description or len(description.strip()) < 15:
            return {
                "error": "Dados insuficientes para análise técnica",
                "trend": "neutro",
                "confidence": 0,
                "processing_time": time.time() - start_time
            }
        
        # Preparar prompt para o modelo
        prompt = f"""
        Realize uma análise técnica com base nos seguintes dados:
        
        ATIVO: {symbol}
        TIMEFRAME: {timeframe}
        
        DADOS: {description}
        
        Forneça sua análise no seguinte formato JSON (sem explicações adicionais):
        {{
            "trend": (string: "bullish", "bearish" ou "neutral"),
            "trend_strength": (número de 0 a 100),
            "key_levels": {{
                "support": [níveis de suporte],
                "resistance": [níveis de resistência]
            }},
            "indicators": [
                {{
                    "name": "nome do indicador",
                    "value": "valor atual",
                    "signal": "bullish/bearish/neutral",
                    "description": "breve descrição"
                }}
            ],
            "patterns": [padrões gráficos identificados],
            "short_term_forecast": "previsão para curto prazo",
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
                result["symbol"] = symbol
                result["timeframe"] = timeframe
                result["data_type"] = data_type
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Falha ao extrair JSON, tentar parsear manualmente
                return self._parse_non_json_response(response, symbol, timeframe, start_time)
                
        except Exception as e:
            return {
                "error": f"Erro na análise técnica: {str(e)}",
                "trend": "neutral",
                "confidence": 0,
                "symbol": symbol,
                "timeframe": timeframe,
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
        
        # Adicionar dados de preço se disponíveis
        if "price_data" in data:
            price_data = data["price_data"]
            
            # Resumo de preços
            if len(price_data) > 0:
                first = price_data[0]
                last = price_data[-1]
                highest = max(price_data, key=lambda x: x.get("high", 0))
                lowest = min(price_data, key=lambda x: x.get("low", float("inf")))
                
                description.append(f"Preço inicial: {first.get('close', 'N/A')}")
                description.append(f"Preço atual: {last.get('close', 'N/A')}")
                description.append(f"Máxima do período: {highest.get('high', 'N/A')}")
                description.append(f"Mínima do período: {lowest.get('low', 'N/A')}")
                
                # Variação percentual
                if "close" in first and "close" in last and first["close"] > 0:
                    change_pct = (last["close"] - first["close"]) / first["close"] * 100
                    description.append(f"Variação no período: {change_pct:.2f}%")
            
            # Adicionar dados de volume se presentes
            if any("volume" in candle for candle in price_data):
                avg_volume = sum(candle.get("volume", 0) for candle in price_data) / len(price_data)
                last_volume = price_data[-1].get("volume", 0)
                
                description.append(f"Volume médio: {avg_volume:.2f}")
                description.append(f"Volume atual: {last_volume}")
                
                # Comparar volume atual com média
                if avg_volume > 0:
                    vol_change = (last_volume - avg_volume) / avg_volume * 100
                    description.append(f"Volume em relação à média: {vol_change:.2f}%")
        
        # Adicionar indicadores pré-calculados se disponíveis
        if "indicators" in data:
            indicators = data["indicators"]
            description.append("\nIndicadores:")
            
            for ind_name, ind_value in indicators.items():
                if isinstance(ind_value, dict):
                    ind_info = [f"{k}: {v}" for k, v in ind_value.items()]
                    description.append(f"- {ind_name}: {', '.join(ind_info)}")
                else:
                    description.append(f"- {ind_name}: {ind_value}")
        
        # Adicionar padrões pré-identificados se disponíveis
        if "patterns" in data:
            patterns = data["patterns"]
            if patterns:
                description.append("\nPadrões identificados:")
                for pattern in patterns:
                    description.append(f"- {pattern}")
        
        return "\n".join(description)
    
    def _parse_non_json_response(self, response: str, symbol: str, timeframe: str, start_time: float) -> Dict[str, Any]:
        """
        Tenta extrair informações de uma resposta que não está em formato JSON
        
        Args:
            response: Texto da resposta
            symbol: Símbolo do ativo
            timeframe: Timeframe da análise
            start_time: Tempo de início do processamento
            
        Returns:
            Dicionário com resultados da análise
        """
        result = {
            "trend": "neutral",
            "trend_strength": 50,
            "key_levels": {
                "support": [],
                "resistance": []
            },
            "indicators": [],
            "patterns": [],
            "short_term_forecast": "Indefinido",
            "confidence": 30,
            "symbol": symbol,
            "timeframe": timeframe,
            "processing_time": time.time() - start_time
        }
        
        # Tentar extrair tendência
        trend_terms = {
            "bullish": ["altista", "bullish", "alta", "positivo", "compra", "subida"],
            "bearish": ["baixista", "bearish", "queda", "negativo", "venda", "descida"]
        }
        
        for trend_type, keywords in trend_terms.items():
            if any(kw in response.lower() for kw in keywords):
                result["trend"] = trend_type
                break
        
        # Tentar extrair força da tendência
        strength_matches = []
        lines = response.split("\n")
        for line in lines:
            if any(term in line.lower() for term in ["força", "strength", "intensidade"]) and ":" in line:
                try:
                    value_str = line.split(":", 1)[1].strip()
                    # Extrair número
                    nums = [int(s) for s in value_str.split() if s.isdigit()]
                    if nums:
                        result["trend_strength"] = min(100, max(0, nums[0]))
                        break
                except:
                    pass
        
        # Tentar extrair níveis de suporte e resistência
        for level_type in ["suporte", "support", "resistência", "resistance"]:
            for line in lines:
                if level_type in line.lower() and ":" in line:
                    try:
                        values_str = line.split(":", 1)[1].strip()
                        # Extrair números
                        import re
                        nums = re.findall(r'\d+\.?\d*', values_str)
                        nums = [float(n) for n in nums]
                        
                        if nums:
                            if level_type in ["suporte", "support"]:
                                result["key_levels"]["support"].extend(nums)
                            else:
                                result["key_levels"]["resistance"].extend(nums)
                    except:
                        pass
        
        # Tentar extrair padrões
        pattern_section = False
        for line in lines:
            if "padrões" in line.lower() or "patterns" in line.lower():
                pattern_section = True
                continue
                
            if pattern_section and (line.startswith("-") or line.startswith("*")):
                pattern = line[1:].strip()
                if pattern:
                    result["patterns"].append(pattern)
        
        # Tentar extrair previsão de curto prazo
        forecast_section = False
        for i, line in enumerate(lines):
            if any(term in line.lower() for term in ["forecast", "previsão", "próximos dias", "curto prazo"]):
                forecast_section = True
                # Pegar a próxima linha não vazia
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                
                if j < len(lines):
                    result["short_term_forecast"] = lines[j].strip()
                    break
        
        return result
    
    def analyze_multi_timeframe(self, symbol: str, data_by_timeframe: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Realiza análise técnica em múltiplos timeframes e consolida os resultados
        
        Args:
            symbol: Símbolo do ativo
            data_by_timeframe: Dicionário com dados para cada timeframe
            
        Returns:
            Análise técnica consolidada entre timeframes
        """
        start_time = time.time()
        
        results = {}
        
        # Analisar cada timeframe individualmente
        for timeframe, data in data_by_timeframe.items():
            data["symbol"] = symbol
            data["timeframe"] = timeframe
            results[timeframe] = self.analyze(data)
        
        # Consolidar resultados
        timeframes_count = len(results)
        
        if timeframes_count == 0:
            return {
                "error": "Nenhum timeframe fornecido para análise",
                "symbol": symbol,
                "processing_time": time.time() - start_time
            }
        
        # Determinar tendência global (ponderando mais os timeframes maiores)
        trend_votes = {"bullish": 0, "bearish": 0, "neutral": 0}
        
        # Pesos para diferentes timeframes (maior peso para timeframes maiores)
        timeframe_weights = {
            "1m": 1, "5m": 1.5, "15m": 2, "30m": 2.5, 
            "1h": 3, "4h": 4, "1d": 5, "1w": 6, "1M": 7
        }
        
        total_weight = 0
        
        for tf, result in results.items():
            weight = timeframe_weights.get(tf, 1)
            total_weight += weight
            trend_votes[result.get("trend", "neutral")] += weight
        
        # Determinar tendência predominante
        global_trend = max(trend_votes.items(), key=lambda x: x[1])[0]
        
        # Calcular força da tendência média ponderada
        trend_strength = 0
        for tf, result in results.items():
            weight = timeframe_weights.get(tf, 1)
            trend_strength += result.get("trend_strength", 50) * weight
        
        if total_weight > 0:
            trend_strength = trend_strength / total_weight
        
        # Consolidar suportes e resistências (eliminar duplicados e valores muito próximos)
        all_supports = []
        all_resistances = []
        
        for tf, result in results.items():
            key_levels = result.get("key_levels", {})
            all_supports.extend(key_levels.get("support", []))
            all_resistances.extend(key_levels.get("resistance", []))
        
        # Filtrar níveis duplicados ou muito próximos (margem de 1%)
        def filter_nearby_levels(levels):
            if not levels:
                return []
            
            levels.sort()
            result = [levels[0]]
            
            for level in levels[1:]:
                last = result[-1]
                # Se o nível estiver a mais de 1% do último, adicioná-lo
                if (level - last) / last > 0.01:
                    result.append(level)
                    
            return result
        
        consolidated_supports = filter_nearby_levels(all_supports)
        consolidated_resistances = filter_nearby_levels(all_resistances)
        
        # Consolidar padrões (manter apenas os que aparecem em múltiplos timeframes)
        pattern_counts = {}
        
        for tf, result in results.items():
            patterns = result.get("patterns", [])
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        significant_patterns = [pattern for pattern, count in pattern_counts.items() 
                               if count > 1 or "1d" in results and pattern in results["1d"].get("patterns", [])]
        
        # Criar resultado consolidado
        consolidated_result = {
            "symbol": symbol,
            "trend": global_trend,
            "trend_strength": round(trend_strength, 1),
            "key_levels": {
                "support": consolidated_supports,
                "resistance": consolidated_resistances
            },
            "patterns": significant_patterns,
            "timeframe_analysis": {tf: {
                "trend": result.get("trend"),
                "trend_strength": result.get("trend_strength"),
                "confidence": result.get("confidence")
            } for tf, result in results.items()},
            "processing_time": time.time() - start_time
        }
        
        # Adicionar previsão de curto prazo do maior timeframe disponível
        for tf in ["1d", "4h", "1h", "30m", "15m", "5m", "1m"]:
            if tf in results and "short_term_forecast" in results[tf]:
                consolidated_result["short_term_forecast"] = results[tf]["short_term_forecast"]
                break
        
        # Calcular confiança global (média ponderada)
        confidence = 0
        for tf, result in results.items():
            weight = timeframe_weights.get(tf, 1)
            confidence += result.get("confidence", 50) * weight
        
        if total_weight > 0:
            consolidated_result["confidence"] = round(confidence / total_weight, 1)
        else:
            consolidated_result["confidence"] = 50
            
        return consolidated_result
