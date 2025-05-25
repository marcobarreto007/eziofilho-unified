"""
Script de teste para o sistema unificado EzioFilho
Testa a funcionalidade da hierarquia de classes refatorada
"""
import logging
import json
import time
import sys
from pathlib import Path

# Adicionar diretório raiz ao sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core.unified_base_expert import EzioBaseExpert
from core.unified_sentiment_expert import SentimentExpert
from core.unified_orchestrator import UnifiedOrchestrator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_unified")

def test_base_expert():
    """Testa a classe base EzioBaseExpert"""
    print("\n=== Teste do EzioBaseExpert Unificado ===")
    
    # Criar configuração JSON básica
    models_config = {
        "models": {
            "sentiment": {
                "path": "microsoft/phi-2",
                "system_message": "Você é um analista financeiro especializado em análise de sentimento de mercado. Avalie o texto fornecido e forneça uma análise detalhada.",
                "temperature": 0.1,
                "quantization": "8bit"
            }
        }
    }
    
    # Salvar configuração em arquivo temporário
    config_path = Path("temp_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(models_config, f, indent=2)
    
    print(f"Configuração salva em {config_path.absolute()}")
    
    try:
        # Inicializar o expert
        print("\nInicializando EzioBaseExpert para teste...")
        expert = EzioBaseExpert(
            expert_type="sentiment",
            config_path=config_path,
            gpu_id=None  # Usar auto-seleção
        )
        
        # Exibir informações do expert
        print("\nInformações do Expert:")
        print(f"Expert Type: {expert.expert_type}")
        print(f"Expert ID: {expert.expert_id}")
        print(f"Device: {expert.device}")
        print(f"Inicializado: {expert.is_initialized}")
        if not expert.is_initialized and expert.initialization_error:
            print(f"Erro de Inicialização: {expert.initialization_error}")
        
        # Exibir status detalhado
        print("\nStatus Detalhado:")
        status = expert.get_status()
        print(json.dumps(status, indent=2))
        
        return expert
        
    except Exception as e:
        print(f"Erro durante teste de EzioBaseExpert: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Remover arquivo de configuração temporário
        if config_path.exists():
            config_path.unlink()

def test_sentiment_expert():
    """Testa o especialista SentimentExpert"""
    print("\n=== Teste do SentimentExpert Unificado ===")
    
    # Criar configuração JSON básica
    models_config = {
        "models": {
            "sentiment": {
                "path": "microsoft/phi-2",
                "system_message": "Você é um analista financeiro especializado em análise de sentimento de mercado. Avalie o texto fornecido e forneça uma análise detalhada.",
                "temperature": 0.1,
                "quantization": "8bit"
            }
        }
    }
    
    # Salvar configuração em arquivo temporário
    config_path = Path("temp_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(models_config, f, indent=2)
    
    print(f"Configuração salva em {config_path.absolute()}")
    
    try:
        # Inicializar o expert
        print("\nInicializando SentimentExpert para teste...")
        expert = SentimentExpert(
            config_path=config_path,
            gpu_id=None  # Usar auto-seleção
        )
        
        # Exibir informações do expert
        print("\nInformações do Expert:")
        print(f"Expert Type: {expert.expert_type}")
        print(f"Expert ID: {expert.expert_id}")
        print(f"Device: {expert.device}")
        print(f"Inicializado: {expert.is_initialized}")
        if not expert.is_initialized and expert.initialization_error:
            print(f"Erro de Inicialização: {expert.initialization_error}")
        
        # Realizar análise de sentimento
        print("\nExecutando análise de sentimento...")
        text = "A Apple anuncia lucros recordes para o primeiro trimestre de 2025, superando as expectativas dos analistas em 12%. As vendas do iPhone cresceram 15% em relação ao ano anterior."
        
        start_time = time.time()
        result = expert.analyze_sentiment(text)
        elapsed_time = time.time() - start_time
        
        # Mostrar resultado
        print(f"\n=== Resultado da Análise (tempo: {elapsed_time:.2f}s) ===")
        print(f"Status: {result.get('status', 'N/A')}")
        
        if result["status"] == "success" and "summary" in result:
            print("\nResumo:")
            print(result["summary"])
        else:
            print(f"Erro: {result.get('error', 'Erro desconhecido')}")
        
        # Exibir métricas
        print("\nMétricas:")
        metrics = expert.get_metrics()
        print(json.dumps(metrics, indent=2))
        
        return expert, result
        
    except Exception as e:
        print(f"Erro durante teste de SentimentExpert: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        # Remover arquivo de configuração temporário
        if config_path.exists():
            config_path.unlink()

def test_orchestrator():
    """Testa o orquestrador unificado"""
    print("\n=== Teste do UnifiedOrchestrator ===")
    
    # Criar configuração JSON básica
    models_config = {
        "models": {
            "sentiment": {
                "path": "microsoft/phi-2",
                "system_message": "Você é um analista financeiro especializado em análise de sentimento de mercado. Avalie o texto fornecido e forneça uma análise detalhada.",
                "temperature": 0.1,
                "quantization": "8bit"
            }
        }
    }
    
    # Salvar configuração em arquivo temporário
    config_path = Path("temp_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(models_config, f, indent=2)
    
    print(f"Configuração salva em {config_path.absolute()}")
    
    try:
        # Inicializar o orquestrador
        print("\nInicializando UnifiedOrchestrator para teste...")
        orchestrator = UnifiedOrchestrator(
            config_path=config_path,
            expert_types=["sentiment"]
        )
        
        # Exibir informações do orquestrador
        print("\nInformações do Orquestrador:")
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
        
        # Verificar se o especialista de sentimento foi inicializado
        if "sentiment" in status["experts"] and status["experts"]["sentiment"]["status"] == "initialized":
            # Realizar análise
            print("\nExecutando análise com o orquestrador...")
            text = "A inflação subiu 0.5% no último trimestre, abaixo das expectativas de 0.8%. O banco central sinalizou que poderá reduzir as taxas de juros nas próximas reuniões."
            
            start_time = time.time()
            result = orchestrator.analyze(text=text, experts=["sentiment"])
            elapsed_time = time.time() - start_time
            
            # Mostrar resultado
            print(f"\n=== Resultado da Análise (tempo: {elapsed_time:.2f}s) ===")
            print(f"Status: {result.get('status', 'N/A')}")
            print("\nResumo:")
            print(result["summary"])
            
            return orchestrator, result
        else:
            print("Especialista de sentimento não foi inicializado corretamente.")
            return orchestrator, None
        
    except Exception as e:
        print(f"Erro durante teste de UnifiedOrchestrator: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        # Remover arquivo de configuração temporário
        if config_path.exists():
            config_path.unlink()

def main():
    """Função principal"""
    print("==================================================")
    print("  TESTE DO SISTEMA EZIOFILHO UNIFICADO")
    print("==================================================\n")
    
    # Verificar disponibilidade de GPU
    import torch
    if torch.cuda.is_available():
        print(f"GPU disponível: {torch.cuda.device_count()} dispositivo(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    Memória total: {mem:.2f} GB")
    else:
        print("GPU não disponível, utilizando CPU")
    
    # Testar componentes
    print("\n1. Testando classe base EzioBaseExpert...")
    base_expert = test_base_expert()
    
    print("\n2. Testando especialista SentimentExpert...")
    sentiment_expert, sentiment_result = test_sentiment_expert()
    
    print("\n3. Testando orquestrador unificado...")
    orchestrator, orchestrator_result = test_orchestrator()
    
    # Resumo dos testes
    print("\n==================================================")
    print("  RESUMO DOS TESTES")
    print("==================================================")
    
    print("\nBase Expert: ", "✅ OK" if base_expert else "❌ Falhou")
    print("Sentiment Expert: ", "✅ OK" if sentiment_expert else "❌ Falhou")
    print("Orchestrator: ", "✅ OK" if orchestrator else "❌ Falhou")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
