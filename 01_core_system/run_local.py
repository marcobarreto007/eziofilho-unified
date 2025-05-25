#!/usr/bin/env python
"""
run_local.py · Carrega um modelo GGUF via llama-cpp-python e
abre um prompt interativo.

❱❱  python run_local.py                   # usa caminho default abaixo
❱❱  python run_local.py --model D:\models\phi-2.Q2_K.gguf
❱❱  MODEL_PATH=D:\models\phi-2.Q2_K.gguf python run_local.py
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from time import perf_counter

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover
    sys.stderr.write("❌  llama-cpp-python não instalado: pip install llama-cpp-python\n")
    raise

# ────────────────────────── CONFIGURAÇÃO PADRÃO ──────────────────────────
DEFAULT_MODELS_DIR = pathlib.Path(
    r"C:\Users\anapa\EzioFilhoUnified\modelos_hf\TheBloke--Phi-2-GGUF"
)
DEFAULT_MODEL_FILE = DEFAULT_MODELS_DIR / "phi-2.Q2_K.gguf"

# ────────────────────────── LOGGING BÁSICO ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_local")

# ────────────────────────── FUNÇÕES AUXILIARES ───────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load local GGUF model and chat")
    parser.add_argument(
        "--model",
        "-m",
        type=pathlib.Path,
        default=os.getenv("MODEL_PATH", DEFAULT_MODEL_FILE),
        help="Caminho completo do arquivo .gguf",
    )
    return parser.parse_args()


def load_llm(model_path: pathlib.Path) -> Llama:
    if not model_path.exists():
        log.error("Modelo não encontrado: %s", model_path)
        sys.exit(1)

    log.info("🔄  Carregando modelo %s … aguarde", model_path.name)
    tic = perf_counter()
    llm = Llama(model_path=str(model_path), n_gpu_layers=0, n_ctx=2048)
    log.info("✅  Modelo carregado em %.1f s", perf_counter() - tic)
    return llm


def main() -> None:
    args = parse_args()
    llm = load_llm(args.model)

    print("\n💬  Digite sua pergunta (Ctrl+C para sair)\n")
    try:
        while True:
            user = input("Você: ").strip()
            if not user:
                continue

            tic = perf_counter()
            response = llm(user, max_tokens=256, temperature=0.7)
            answer = response["choices"][0]["text"].strip()
            print(f"🤖 {answer}  ⏱️ {perf_counter() - tic:.1f}s\n")

    except KeyboardInterrupt:
        print("\n👋  Até a próxima!")
        sys.exit(0)


if __name__ == "__main__":
    main()
