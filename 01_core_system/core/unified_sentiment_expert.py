"""
SentimentExpert - Especialista unificado em análise de sentimento financeiro
Versão 3.0.0 - Baseada na nova implementação unificada EzioBaseExpert
"""

import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, field

# Importar a classe base unificada
from core.unified_base_expert import EzioBaseExpert

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SentimentExpert")

# Decorador para rastrear tempo de execução
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
    """Classe de dados para resultados estruturados de análise de sentimento"""
    score: float  # Pontuação entre -5 (muito negativo) e +5 (muito positivo)
    classification: str  # Classificação textual (muito negativo, negativo, neutro, positivo, muito positivo)
    confidence: str  # Nível de confiança (baixo, médio, alto)
    emoji: str  # Emoji representativo do sentimento
    risk_level: str  # Nível de risco (baixo, moderado, alto)
    volatility_expectation: str  # Expectativa de volatilidade (baixa, moderada, alta)
    key_factors: List[str] = field(default_factory=list)  # Fatores-chave identificados
    market_implications: List[str] = field(default_factory=list)  # Implicações para o mercado
    evidence: List[str] = field(default_factory=list)  # Evidências textuais
    recommended_action: str = ""  # Ação recomendada
    analysis_timestamp: datetime = field(default_factory=datetime.now)  # Timestamp da análise

    def as_dict(self) -> Dict[str, Any]:
        """Converte o resultado para dicionário"""
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
        """Gera resumo formatado da análise de sentimento"""
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
            lines.append("\nImplicações para o Mercado:")
            lines.extend(f"• {m}" for m in self.market_implications)
            
        if self.evidence:
            lines.append("\nEvidências:")
            lines.extend(f"• {e}" for e in self.evidence)
            
        lines.append(f"\nAnálise realizada em: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

class SentimentExpert(EzioBaseExpert):
    """Especialista em análise de sentimento financeiro com suporte avançado a LLM"""
    
    # Versão específica deste especialista
    VERSION = "3.0.0"

    # Prompt padrão para análise de sentimento
    SYSTEM_PROMPT = (
        "Você é um especialista em análise de sentimento financeiro. Analise o texto abaixo e retorne um JSON com:"
        " score (-5 a +5), confidence (low|medium|high), key_factors (2-5), market_implications (2-4),"
        " evidence (citações), recommended_action."
        " Foque apenas no sentimento financeiro e implicações de mercado."
    )

    def __init__(self, 
                config_path: Optional[Path] = None, 
                gpu_id: Optional[int] = None,
                model_path_override: Optional[str] = None,
                quantization: Optional[str] = None):
        """
        Inicializa o especialista em sentimento
        
        Args:
            config_path: Caminho para arquivo de configuração
            gpu_id: ID da GPU a utilizar
            model_path_override: Sobrescrição do caminho do modelo
            quantization: Método de quantização
        """
        self.initialization_time = time.time()
        
        # Inicializar classe base
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
        self.metrics["json_extraction_failures"] = 0
        self.metrics["processing_failures"] = 0

    @time_execution
    def analyze_sentiment(self, text: str, context: Optional[str] = None, include_raw_response: bool = False) -> Dict[str, Any]:
        """
        Analisa sentimento financeiro no texto com saída estruturada
        
        Args:
            text: O texto a ser analisado
            context: Contexto adicional (opcional)
            include_raw_response: Incluir resposta bruta do LLM
            
        Returns:
            Resultado estruturado da análise
        """
        self.metrics["sentiment_calls"] += 1
        
        # Executar análise usando o modelo base
        raw_result = self.analyze(
            text=text,
            context=context,
            max_tokens=512,
            temperature=0.1
        )
        
        # Verificar se foi bem-sucedido
        if raw_result["status"] != "success":
            return {
                "status": "error",
                "error": raw_result.get("error", "Análise falhou"),
                "sentiment": None
            }
        
        # Processar resposta bruta em formato estruturado
        try:
            sentiment_result = self._process_response(text, raw_result["response"])
            
            result = {
                "status": "success",
                "sentiment": sentiment_result.as_dict(),
                "summary": sentiment_result.summary()
            }
            
            # Adicionar resposta bruta se solicitado
            if include_raw_response:
                result["raw_response"] = raw_result["response"]
                
            return result
        
        except Exception as e:
            logger.error(f"Erro processando resposta de sentimento: {e}")
            self.metrics["processing_failures"] += 1
            return {
                "status": "error",
                "error": f"Erro de processamento: {str(e)}",
                "sentiment": None
            }

    @time_execution
    def process_query(self, query: str, context: Dict[str, Any] = None) -> Tuple[Dict[str, Any], float]:
        """
        Processa consulta de sentimento usando modelo
        
        Args:
            query: A consulta ou texto a analisar
            context: Contexto adicional
            
        Returns:
            Tupla com (resultado, confiança)
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
        """
        Extrai análise estruturada de sentimento da resposta do modelo
        
        Args:
            original_text: Texto original analisado
            response: Resposta do modelo
            
        Returns:
            Objeto SentimentResult preenchido
        """
        # Tentar extrair JSON da resposta
        try:
            # Procurar bloco JSON na resposta
            json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', response)
            data = {}
            
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
            else:
                # Tentar extrair JSON sem bloco de código
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
            
            # Extrair dados do JSON
            score = data.get("score", 0)
            # Converter para escala de -5 a +5 se necessário
            if score is not None and abs(score) <= 1.0:
                score *= 5
            
            classification = data.get("classification", "")
            if not classification:
                # Derivar classificação da pontuação
                if score > 3:
                    classification = "muito positivo"
                elif score > 0:
                    classification = "positivo"
                elif score == 0:
                    classification = "neutro"
                elif score > -3:
                    classification = "negativo"
                else:
                    classification = "muito negativo"
            
            # Obter ou derivar outros campos
            confidence = data.get("confidence", "medium")
            key_factors = data.get("key_factors", [])
            market_implications = data.get("market_implications", [])
            evidence = data.get("evidence", [])
            recommended_action = data.get("recommended_action", "")
            
            # Definir emoji com base na pontuação
            emoji = "📈" if score > 2 else "↗️" if score > 0 else "➖" if score == 0 else "↘️" if score > -2 else "📉"
            
            return SentimentResult(
                score=score,
                classification=classification,
                confidence=confidence,
                emoji=emoji,
                risk_level="alto" if abs(score) > 3 else "moderado" if abs(score) > 1 else "baixo",
                volatility_expectation="alta" if abs(score) > 3 else "moderada" if abs(score) > 1 else "baixa",
                key_factors=key_factors,
                market_implications=market_implications,
                evidence=evidence,
                recommended_action=recommended_action
            )
            
        except Exception as e:
            # Fallback para análise simplificada se a extração do JSON falhar
            logger.warning(f"Falha na extração de JSON da resposta: {e}. Usando análise simplificada.")
            self.metrics["json_extraction_failures"] += 1
            
            # Análise simplificada baseada no texto original
            score = round((original_text.lower().count("up") - original_text.lower().count("down")) * 1.2, 2)
            score = max(-5, min(5, score))  # Limitar entre -5 e 5
            
            classification = "muito positivo" if score > 3 else "positivo" if score > 0 else "neutro" if score == 0 else "negativo" if score > -3 else "muito negativo"
            emoji = "📈" if score > 2 else "↗️" if score > 0 else "➖" if score == 0 else "↘️" if score > -2 else "📉"
            
            return SentimentResult(
                score=score,
                classification=classification,
                confidence="low",  # Baixa confiança pois é fallback
                emoji=emoji,
                risk_level="moderado",
                volatility_expectation="moderada",
                key_factors=["Análise simplificada"],
                market_implications=["Avaliação limitada disponível"] if score > 0 else ["Cautela recomendada"],
                evidence=["Análise baseada em contagem de palavras-chave"],
                recommended_action="Buscar análise complementar"
            )

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna informações de status do especialista
        
        Returns:
            Dicionário com informações de status
        """
        # Obter status base
        status = super().get_status()
        
        # Adicionar informações específicas do especialista de sentimento
        status["sentiment_expert_metrics"] = {
            "calls": self.metrics.get("calls", 0),
            "sentiment_calls": self.metrics.get("sentiment_calls", 0),
            "json_extraction_failures": self.metrics.get("json_extraction_failures", 0),
            "processing_failures": self.metrics.get("processing_failures", 0)
        }
        
        status["version"] = self.VERSION
            
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
                "expert_id": self.expert_id,
                "expert_version": self.VERSION
            }
            
            # Salvar como JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                
            return output_path
                
        except Exception as e:
            logger.error(f"Erro ao salvar resultado: {e}")
            return None
