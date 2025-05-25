# Caminho: C:\Users\anapa\SuperIA\EzioFilhoUnified\experts_fingpt\expert_fingpt.py

"""
ExpertFinGPT - Wrapper entre Ezio e o módulo FinGPT.
Lê entrada.json, executa o modelo FinGPT (Sentiment Analysis v1), salva resposta em saida_fingpt.json.
"""

import json
import logging
import sys
from pathlib import Path

# Configuração de logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ExpertFinGPT")

# Caminhos de arquivos
ENTRADA_PATH = Path("C:/Users/anapa/SuperIA/EzioFilhoUnified/entrada.json")
SAIDA_PATH = Path("C:/Users/anapa/SuperIA/EzioFilhoUnified/saida_fingpt.json")
FINGPT_MOD_PATH = Path("C:/Users/anapa/FinGPT")
FINGPT_LIB = FINGPT_MOD_PATH / "fingpt"

# 1. Diagnóstico: Mostra se o caminho do módulo existe e printa sys.path real
if not FINGPT_LIB.exists():
    logger.error(f"❌ Diretório não encontrado: {FINGPT_LIB}")
    sys.exit(1)
else:
    logger.info(f"✅ Diretório do FinGPT localizado em: {FINGPT_LIB.resolve()}")

if str(FINGPT_LIB.resolve()) not in sys.path:
    sys.path.insert(0, str(FINGPT_LIB.resolve()))
logger.info("sys.path em uso:\n%s", "\n".join(sys.path[:3]))

# 2. Importação robusta do modelo (diagnóstico extra)
try:
    from FinGPT_Sentiment_Analysis_v1.inferencing.infer import main as run_fingpt_inference
    logger.info("✅ Importação da função principal do FinGPT concluída com sucesso.")
except ImportError as e:
    logger.error("❌ Erro ao importar função do FinGPT Sentiment Analysis v1: %s", e)
    raise RuntimeError("Falha crítica: não foi possível importar o modelo FinGPT") from e

def carregar_prompt() -> str:
    """Carrega prompt do arquivo entrada.json"""
    if not ENTRADA_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ENTRADA_PATH}")
    with open(ENTRADA_PATH, "r", encoding="utf-8") as f:
        try:
            dados = json.load(f)
            prompt = dados.get("prompt", "").strip()
            if not prompt:
                raise ValueError("O campo 'prompt' está vazio.")
            return prompt
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao decodificar JSON: {e}")

def salvar_saida(dados: dict):
    """Salva resposta final em saida_fingpt.json"""
    try:
        with open(SAIDA_PATH, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Resultado salvo em: {SAIDA_PATH}")
    except Exception as e:
        logger.error(f"Erro ao salvar arquivo de saída: {e}")
        raise

def executar_expert():
    """Executa o fluxo completo do Expert FinGPT"""
    try:
        logger.info("🚀 Iniciando execução do Expert FinGPT")
        # 1. Carrega o prompt
        prompt = carregar_prompt()
        logger.info(f"📥 Prompt recebido:\n{prompt}")

        # 2. Chama o modelo FinGPT
        resposta = run_fingpt_inference(prompt)
        logger.info(f"📤 Resposta gerada:\n{resposta}")

        # 3. Monta estrutura de saída
        saida = {
            "expert": "FinGPT",
            "input": prompt,
            "output": resposta,
            "score": 1.0
        }

        # 4. Salva saída
        salvar_saida(saida)

    except Exception as e:
        erro_msg = f"❌ Erro na execução: {str(e)}"
        logger.error(erro_msg)
        salvar_saida({
            "expert": "FinGPT",
            "input": "",
            "output": erro_msg,
            "score": 0.0
        })

if __name__ == "__main__":
    executar_expert()
