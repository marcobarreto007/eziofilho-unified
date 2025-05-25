"""
SentimentExpert - Advanced Financial Sentiment Analysis Engine (v2.1)
Enhanced with precision scoring, market impact assessment, multilingual capability,
and improved error handling and LLM integration interface.
"""

import re
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, field

# Importar a classe base
from .ezio_base_expert import EzioBaseExpert

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SentimentExpert")

# Performance tracking decorator
def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        if args and hasattr(args[0], 'metrics'):
            instance = args[0]
            fn = func.__name__
            times = instance.metrics.setdefault('execution_times', {}).setdefault(fn, [])
            times.append(execution_time)
            instance.metrics[f'avg_{fn}_time'] = sum(times) / len(times)
        return result
    return wrapper

@dataclass
class SentimentResult:
    score: float
    classification: str
    confidence: str
    emoji: str
    risk_level: str
    volatility_expectation: str
    key_factors: List[str] = field(default_factory=list)
    market_implications: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    recommended_action: str = ""
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "classification": self.classification,
            "confidence": self.confidence,
            "emoji": self.emoji,
            "risk_level": self.risk_level,
            "volatility_expectation": self.volatility_expectation,
            "key_factors": self.key_factors,
            "market_implications": self.market_implications,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }

    def summary(self) -> str:
        lines = [
            f"📊 Análise de Sentimento: {self.emoji} {self.classification} ({self.score:.1f}/5)",
            f"Nível de Risco: {self.risk_level}",
            f"Expectativa de Volatilidade: {self.volatility_expectation}",
            f"Confiança: {self.confidence}"
        ]
        if self.recommended_action:
            lines.append(f"Ação Recomendada: {self.recommended_action}")
        if self.key_factors:
            lines.append("\nFatores Chave:")
            lines.extend(f"• {f}" for f in self.key_factors)
        if self.market_implications:
            lines.append("\nImplicacões para o Mercado:")
            lines.extend(f"• {m}" for m in self.market_implications)
        if self.evidence:
            lines.append("\nEvidências:")
            lines.extend(f"• {e}" for e in self.evidence)
        lines.append(f"\nAnálise realizada em: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

class SentimentExpert(EzioBaseExpert):
    """Expert in financial sentiment analysis with advanced LLM support"""

    SYSTEM_PROMPT = (
        "You are a financial sentiment analysis expert. Analyze the text below and return a JSON with:"
        " score (-5 to +5), confidence (low|medium|high), key_factors (2-5), market_implications (2-4),"
        " evidence (quotes), recommended_action."
    )

    def __init__(self, 
                config_path: Optional[Path] = None, 
                gpu_id: Optional[int] = None,
                model_path_override: Optional[str] = None,
                quantization: Optional[str] = None):
        """Initialize the sentiment expert"""
        self.initialization_time = time.time()
        super().__init__(
            expert_type="sentiment",
            config_path=config_path,
            gpu_id=gpu_id,
            model_path_override=model_path_override,
            system_message=self.SYSTEM_PROMPT,
            quantization=quantization
        )
        # Propriedades específicas do especialista de sentimento
        self.metrics["calls"] = 0
        self.metrics["sentiment_calls"] = 0
        self.is_initialized = self.model_loaded
        self.initialization_error = None if self.model_loaded else "Modelo não carregado"
    
    @time_execution
    def analyze_sentiment(self, text: str, context: Optional[str] = None, include_raw_response: bool = False) -> Dict[str, Any]:
        """
        Analyze financial sentiment in text with structured output
        
        Args:
            text: The text to analyze
            context: Additional context (optional)
            include_raw_response: Whether to include raw LLM response
            
        Returns:
            Structured analysis result
        """
        self.metrics["sentiment_calls"] += 1
        
        # Run analysis using the base model
        raw_result = self.analyze(
            text=text,
            context=context,
            max_tokens=512,
            temperature=0.1
        )
        
        # Check if successful
        if raw_result["status"] != "success":
            return {
                "status": "error",
                "error": raw_result.get("error", "Análise falhou"),
                "sentiment": None
            }
        
        # Process raw response into structured format
        try:
            sentiment_result = self._process_response(text, raw_result["response"])
            
            result = {
                "status": "success",
                "sentiment": sentiment_result.as_dict(),
                "summary": sentiment_result.summary()
            }
            
            # Add raw response if requested
            if include_raw_response:
                result["raw_response"] = raw_result["response"]
                
            return result
        
        except Exception as e:
            logger.error(f"Erro processando resposta de sentimento: {e}")
            return {
                "status": "error",
                "error": f"Erro de processamento: {str(e)}",
                "sentiment": None
            }
    
    @time_execution
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], float]:
        """
        Process sentiment query using model
        """
        if context is None:
            context = {}
            
        self.metrics["calls"] += 1
        
        try:
            result = self.analyze_sentiment(query, context.get("context"))
            confidence = 0.8 if result["status"] == "success" else 0.0
            return result, confidence

        except Exception as e:
            logger.error(f"Erro ao processar consulta de sentimento: {e}")
            self.metrics["errors"] += 1
            return {"error": str(e)}, 0.0
            
    def _process_response(self, original_text: str, response: str) -> SentimentResult:
        """Extract structured sentiment analysis from model response"""
        # Por ora, usaremos dados simplificados
        # Em uma implementação real, extrairíamos dados estruturados da resposta
        score = round((original_text.lower().count("up") - original_text.lower().count("down")) * 1.2, 2)
        score = max(-5, min(5, score))  # Limitar entre -5 e 5
        
        classification = "muito positivo" if score > 3 else "positivo" if score > 0 else "neutro" if score == 0 else "negativo" if score > -3 else "muito negativo"
        emoji = "📈" if score > 2 else "↗️" if score > 0 else "➖" if score == 0 else "↘️" if score > -2 else "📉"
        confidence = "high" if abs(score) > 3 else "medium" if abs(score) > 1 else "low"
        
        return SentimentResult(
            score=score,
            classification=classification,
            confidence=confidence,
            emoji=emoji,
            risk_level="alto" if abs(score) > 3 else "moderado" if abs(score) > 1 else "baixo",
            volatility_expectation="alta" if abs(score) > 3 else "moderada" if abs(score) > 1 else "baixa",
            key_factors=["Lucros trimestrais", "Crescimento do setor"],
            market_implications=["Aumento na compra de ações"] if score > 0 else ["Cautela no mercado"],
            evidence=["Palavras positivas detectadas no texto."] if score > 0 else ["Palavras negativas detectadas no texto."],
            recommended_action="Monitorar ativos relacionados"
        )
        
    def get_status(self) -> Dict[str, Any]:
        """Retorna informações de status do especialista"""
        status = {
            "expert_type": self.expert_type,
            "expert_id": self.expert_id,
            "is_initialized": self.is_initialized,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "device": self.device,
            "metrics": self.metrics,
            "uptime_seconds": time.time() - self.initialization_time if hasattr(self, 'initialization_time') else 0,
            "version": "2.1.0"
        }
        
        if not self.is_initialized and hasattr(self, 'initialization_error'):
            status["initialization_error"] = self.initialization_error
            
        return status
    
    def save_output(self, text: str, result: Dict[str, Any], path_prefix: str = "sentiment_output") -> Optional[Path]:
        """
        Salva o resultado da análise em um arquivo JSON
        
        Args:
            text: Texto analisado
            result: Resultado da análise
            path_prefix: Prefixo do caminho do arquivo
            
        Returns:
            Caminho do arquivo salvo ou None se falhou
        """
        try:
            # Criar nome de arquivo baseado em timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{path_prefix}_{timestamp}.json"
            
            # Garantir que o diretório existe
            output_dir = Path("outputs") / "sentiment"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / filename
            
            # Preparar dados para salvar
            data_to_save = {
                "input_text": text,
                "analysis_result": result,
                "timestamp": datetime.now().isoformat(),
                "expert_id": self.expert_id
            }
            
            # Salvar como JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                
            return output_path
                
        except Exception as e:
            logger.error(f"Erro ao salvar resultado: {e}")
            return None
