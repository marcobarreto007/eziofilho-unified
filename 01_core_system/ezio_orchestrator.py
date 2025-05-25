"""
Orquestrador para sistema de análise financeira EzioFilho
Gerencia múltiplos especialistas e combina suas análises
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Garantir que o diretório atual está no path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importar especialistas
from ezio_experts import get_expert, get_available_experts

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("EzioOrchestrator")

class EzioOrchestrator:
    """
    Coordenador do sistema de análises financeiras EzioFilho
    Gerencia os especialistas e combina suas análises
    """
    
    # Versão do orquestrador
    VERSION = "2.0.0"
    
    def __init__(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        gpu_id: Optional[int] = None,
        expert_types: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Inicializa o orquestrador com especialistas
        
        Args:
            config_path: Caminho para arquivo de configuração
            gpu_id: ID específico da GPU a ser usada (None para auto-seleção)
            expert_types: Lista de tipos de especialistas a inicializar (None para todos)
            output_dir: Diretório para salvar resultados
        """
        self.initialization_time = time.time()
        self.config_path = Path(config_path) if config_path else Path("models_config.json")
        self.gpu_id = gpu_id
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Inicializar especialistas
        self.experts = {}
        self.expert_types = expert_types or ["sentiment", "technical", "fundamental", "macro"]
        self.expert_status = {expert_type: "not_initialized" for expert_type in self.expert_types}
        
        # Carregar configuração global
        self.config = self._load_global_config()
        
        # Inicializar especialistas
        self._initialize_experts()
        
        logger.info(f"Orquestrador inicializado com {len(self.experts)} especialistas")
    
    def _load_global_config(self) -> Dict[str, Any]:
        """Carregar configuração global"""
        config = {"global_defaults": {}}
        
        if not self.config_path.exists():
            logger.warning(f"Arquivo de configuração não encontrado: {self.config_path}")
            return config
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
            
            # Extrair configuração global
            if "global_defaults" in full_config:
                config["global_defaults"] = full_config["global_defaults"]
                
            # Extrair metadados
            if "metadata" in full_config:
                config["metadata"] = full_config["metadata"]
                
            return config
        except Exception as e:
            logger.error(f"Erro ao carregar configuração global: {e}")
            return config
    
    def _initialize_experts(self) -> None:
        """Inicializar especialistas solicitados"""
        available_experts = get_available_experts()
        
        for expert_type in self.expert_types:
            if expert_type not in available_experts:
                logger.warning(f"Especialista '{expert_type}' não disponível")
                self.expert_status[expert_type] = "not_available"
                continue
                
            try:
                logger.info(f"Inicializando especialista: {expert_type}")
                expert = get_expert(
                    expert_type=expert_type,
                    config_path=self.config_path,
                    gpu_id=self.gpu_id
                )
                
                if expert.is_initialized:
                    self.experts[expert_type] = expert
                    self.expert_status[expert_type] = "initialized"
                    logger.info(f"Especialista {expert_type} inicializado com sucesso")
                else:
                    self.expert_status[expert_type] = f"error: {expert.initialization_error}"
                    logger.error(f"Falha ao inicializar especialista {expert_type}: {expert.initialization_error}")
            except Exception as e:
                self.expert_status[expert_type] = f"error: {str(e)}"
                logger.error(f"Erro ao criar especialista {expert_type}: {e}")
    
    def analyze(
        self, 
        text: str, 
        experts: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analisa texto com os especialistas especificados
        
        Args:
            text: Texto para análise
            experts: Lista de especialistas para usar (None para todos disponíveis)
            output_file: Nome do arquivo para salvar resultado
            context: Contexto adicional para análise
            
        Returns:
            Resultado combinado da análise
        """
        # Determinar especialistas a usar
        expert_types = experts or list(self.experts.keys())
        available_experts = set(self.experts.keys())
        experts_to_use = [e for e in expert_types if e in available_experts]
        
        if not experts_to_use:
            return {
                "status": "error",
                "error": "Nenhum especialista disponível para análise",
                "text": text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Registrar início da análise
        start_time = time.time()
        logger.info(f"Analisando texto com {len(experts_to_use)} especialistas: {', '.join(experts_to_use)}")
        
        # Executar análise com cada especialista
        results = {}
        errors = []
        
        for expert_type in experts_to_use:
            logger.info(f"Executando análise com especialista: {expert_type}")
            expert = self.experts[expert_type]
            
            try:
                # Usar método especializado se disponível
                if expert_type == "sentiment" and hasattr(expert, "analyze_sentiment"):
                    result = expert.analyze_sentiment(text, context=context)
                else:
                    result = expert.analyze(text, context=context)
                
                results[expert_type] = result
                
                if result["status"] == "error":
                    errors.append(f"{expert_type}: {result.get('error', 'Erro desconhecido')}")
            except Exception as e:
                logger.error(f"Erro ao analisar com especialista {expert_type}: {e}")
                errors.append(f"{expert_type}: {str(e)}")
        
        # Gerar análise combinada
        combined_result = {
            "status": "partial_success" if errors else "success",
            "text": text,
            "analysis": results,
            "errors": errors,
            "experts_used": experts_to_use,
            "processing_time": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Criar sumário combinado
        combined_result["summary"] = self._create_summary(combined_result)
        
        # Salvar resultado se solicitado
        if output_file:
            self._save_result(combined_result, output_file)
            
        return combined_result
    
    def _create_summary(self, result: Dict[str, Any]) -> str:
        """Cria resumo combinado de todas as análises"""
        summary = "# Análise Financeira Combinada\n\n"
        
        # Adicionar cabeçalho com timestamp
        summary += f"*Análise gerada em: {result['timestamp']}*\n\n"
        
        # Adicionar análise de sentimento se disponível
        if "sentiment" in result["analysis"]:
            sentiment = result["analysis"]["sentiment"]
            if sentiment["status"] == "success" and "sentiment" in sentiment:
                sent_data = sentiment["sentiment"]
                score = sent_data.get("score")
                classification = sent_data.get("classification", "Não classificado")
                emoji = sent_data.get("emoji", "❓")
                
                summary += f"## 📊 Análise de Sentimento: {emoji} {classification} ({score:.1f}/5)\n\n"
                
                # Adicionar fatores-chave
                if "key_factors" in sent_data:
                    summary += "### Fatores-Chave:\n"
                    for factor in sent_data["key_factors"]:
                        summary += f"* {factor}\n"
                    summary += "\n"
                
                # Adicionar implicações
                if "market_implications" in sent_data:
                    summary += "### Implicações de Mercado:\n"
                    for imp in sent_data["market_implications"]:
                        summary += f"* {imp}\n"
                    summary += "\n"
        
        # Adicionar outras análises conforme implementadas
        
        # Adicionar erros se ocorreram
        if result["errors"]:
            summary += "## ⚠️ Erros Durante Análise\n\n"
            for error in result["errors"]:
                summary += f"* {error}\n"
        
        return summary
    
    def _save_result(self, result: Dict[str, Any], filename: str) -> Optional[Path]:
        """Salva resultado em arquivo"""
        try:
            # Garantir extensão correta
            if not filename.endswith(".json"):
                filename += ".json"
                
            # Caminho completo do arquivo
            file_path = self.output_dir / filename
            
            # Salvar arquivo
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Resultado salvo em: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Erro ao salvar resultado: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna informações de status do orquestrador"""
        return {
            "version": self.VERSION,
            "uptime_seconds": time.time() - self.initialization_time,
            "experts": {
                expert_type: {
                    "status": self.expert_status[expert_type],
                    "details": self.experts[expert_type].get_status() if expert_type in self.experts else None
                }
                for expert_type in self.expert_types
            },
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir)
        }

def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(description="Sistema de Análise Financeira EzioFilho")
    parser.add_argument("--text", type=str, help="Texto para análise")
    parser.add_argument("--file", type=str, help="Arquivo de texto para análise")
    parser.add_argument("--experts", type=str, default="sentiment", 
                       help="Especialistas a usar (separados por vírgula)")
    parser.add_argument("--gpu", type=int, default=None, help="ID da GPU a usar")
    parser.add_argument("--config", type=str, default="models_config.json", 
                       help="Caminho para arquivo de configuração")
    parser.add_argument("--output", type=str, help="Arquivo para salvar resultado")
    parser.add_argument("--status", action="store_true", help="Mostrar status do sistema")
    args = parser.parse_args()
    
    # Inicializar orquestrador
    orchestrator = EzioOrchestrator(
        config_path=args.config,
        gpu_id=args.gpu,
        expert_types=args.experts.split(",") if args.experts else None
    )
    
    # Mostrar status se solicitado
    if args.status:
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
        return
    
    # Verificar se temos texto para análise
    if not args.text and not args.file:
        print("Erro: Forneça texto para análise com --text ou --file")
        return
    
    # Obter texto de arquivo se especificado
    text = args.text
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Erro ao ler arquivo: {e}")
            return
    
    # Executar análise
    result = orchestrator.analyze(
        text=text,
        experts=args.experts.split(",") if args.experts else None,
        output_file=args.output
    )
    
    # Mostrar resultado
    print("\n" + "=" * 80)
    print(" ANÁLISE FINANCEIRA EZIOFILHO ".center(80, "="))
    print("=" * 80 + "\n")
    
    print(result["summary"])
    
    print("\n" + "=" * 80)
    print(f"Análise concluída em {result['processing_time']:.2f} segundos")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()