# Caminho: C:\Users\anapa\SuperIA\EzioFilhoUnified\testes_autogen\teste_modelo_simples.py

import sys
import logging
from pathlib import Path

# Configuração de log simples
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("TesteModeloSimples")

# Adiciona o diretório raiz ao sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.model_router import create_model_router

# Caminho real do modelo GGUF
MODEL_PATH = r"C:\Users\anapa\EzioFilhoUnified\modelos_hf\TheBloke--Phi-2-GGUF\phi-2.Q2_K.gguf"

# Verifica se o arquivo do modelo existe
model_file = Path(MODEL_PATH)
if not model_file.exists():
    logger.error(f"Modelo não encontrado: {MODEL_PATH}")
    sys.exit(1)

# Configuração do modelo para o roteador
model_configs = [
    {
        "name": "phi2",
        "path": MODEL_PATH,
        "model_type": "gguf",
        "capabilities": ["fast", "general"],
    }
]

# Cria roteador e envia prompt
try:
    router = create_model_router(model_configs=model_configs, default_model="phi2")
    prompt = "Explique resumidamente o que é inteligência artificial."
    resposta = router.generate(prompt)
    print("\n📌 Resposta do modelo:")
    print(resposta)
except Exception as e:
    logger.exception(f"Erro durante execução do teste: {e}")
    sys.exit(1)
