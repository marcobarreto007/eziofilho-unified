# Caminho: teste_modelo_mistral.py

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("TesteModeloMistral")

sys.path.append(str(Path(__file__).parent.parent))
from core.model_router import create_model_router

model_path = r"C:\Users\anapa\EzioFilhoUnified\modelos_hf\TheBloke--Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q2_K.gguf"

if not Path(model_path).exists():
    logger.error(f"Modelo não encontrado: {model_path}")
    sys.exit(1)

model_configs = [
    {
        "name": "mistral",
        "path": model_path,
        "model_type": "gguf",
        "capabilities": ["precise", "general"],
    }
]

try:
    router = create_model_router(model_configs=model_configs, default_model="mistral")
    prompt = "Resuma o conceito de aprendizado de máquina em 2 frases."
    resposta = router.generate(prompt)
    print("\n📌 Resposta do modelo:")
    print(resposta)
except Exception as e:
    logger.exception(f"Erro no teste do Mistral: {e}")
    sys.exit(1)
