"""
Script de teste para o EzioBaseExpert
"""
import logging
import json
from pathlib import Path
from ezio_experts.ezio_base_expert import EzioBaseExpert

# Configurar logging
logging.basicConfig(level=logging.INFO)

def main():
    print("=== Teste do EzioBaseExpert ===")
    
    # Criar configuration JSON básica
    models_config = {
        "sentiment": {
            "path": "C:\\Users\\anapa\\EzioFilhoUnified\\modelos_hf\\microsoft--phi-2",
            "system_message": "Você é um analista financeiro especializado em análise de sentimento de mercado. Avalie o texto fornecido e forneça uma análise detalhada.",
            "temperature": 0.1,
            "quantization": "4bit",
            "capabilities": ["financial", "analysis"]
        }
    }
    
    # Salvar configuração em arquivo
    config_path = Path("models_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(models_config, f, indent=2)
    
    print(f"Configuração salva em {config_path.absolute()}")
    
    try:
        # Inicializar o expert (use gpu_id=1 para GTX 1070, ou 0 para RTX 2060)
        print("\nInicializando EzioBaseExpert para análise de sentimento...")
        expert = EzioBaseExpert(
            expert_type="sentiment",
            config_path=config_path,
            gpu_id=1  # Use 1 para GTX 1070, ou 0 para RTX 2060
        )
        
        # Realizar análise
        print("\nExecutando análise de sentimento...")
        prompt = "A Apple anuncia lucros recordes para o primeiro trimestre de 2025, superando as expectativas dos analistas em 12%. As vendas do iPhone cresceram 15% em relação ao ano anterior."
        
        result = expert.analyze(
            prompt=prompt,
            max_tokens=256
        )
        
        # Mostrar resultado
        print("\n=== Resultado da Análise ===")
        print(f"Status: {result.get('status', 'N/A')}")
        print(f"Resposta: {result.get('response', 'N/A')}")
        print("\n=== Métricas ===")
        for key, value in result.get("metrics", {}).items():
            print(f"{key}: {value}")
        
        # Salvar resultado
        output_path = expert.save_output(prompt, result)
        print(f"\nResultado salvo em: {output_path}")
        
    except Exception as e:
        print(f"Erro durante o teste: {e}")

if __name__ == "__main__":
    main()