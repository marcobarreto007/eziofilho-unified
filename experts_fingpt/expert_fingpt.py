"""
ExpertFinGPT – roda Sentiment Analysis v1 e grava saída JSON.
"""
import json, logging, sys, traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FIN_DIR  = PROJECT / "FinGPT"
sys.path.insert(0, str(FIN_DIR))           # torna 'fingpt' importável

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("ExpertFinGPT")

try:
    from fingpt.FinGPT_Sentiment_Analysis_v1.inferencing.infer import main as run_fingpt
except Exception as e:
    log.error("Falha no import:\n%s", traceback.format_exc())
    sys.exit(1)

IN_FILE  = PROJECT / "entrada.json"
OUT_FILE = PROJECT / "saida_fingpt.json"

def carregar_prompt() -> str:
    data = json.loads(IN_FILE.read_text(encoding="utf-8"))
    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise ValueError("Campo 'prompt' vazio!")
    return prompt

def salvar(resposta: str):
    OUT_FILE.write_text(json.dumps({"output": resposta}, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("✅ Resultado salvo em %s", OUT_FILE)

def main():
    try:
        prompt = carregar_prompt()
        resp   = run_fingpt(prompt)
        salvar(resp)
    except Exception:
        log.error("Erro na execução:\n%s", traceback.format_exc())
        salvar("ERRO – veja log")

if __name__ == "__main__":
    main()