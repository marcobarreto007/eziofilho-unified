"""
nlweb_endpoints.py

Micro-serviço FastAPI para geração de explicações em linguagem natural
sobre decisões do sistema EzioFilho, integrando logs via MCP (Magentic-UI)
e inferência em LLM (Claude 3.7 ou outro modelo configurado).
"""

import os
import logging
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------------------------------------------
# 1) Carregar configuração de ambiente
# ---------------------------------------------------
load_dotenv()  # lê .env na raiz, se existir

MAGENTIC_UI_URL = os.getenv("MAGENTIC_UI_URL", "http://localhost:5000")
MAGENTIC_UI_TOKEN = os.getenv("MAGENTIC_UI_TOKEN", "")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.github.com/models/inference/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# ---------------------------------------------------
# 2) Configurar logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("nlweb_explain")

# ---------------------------------------------------
# 3) Inicializar FastAPI
# ---------------------------------------------------
app = FastAPI(
    title="EzioFilho NLWeb Explain Service",
    version="1.0.0",
    description="Exposes /explain to generate natural-language explanations "
                "for financial decisions based on Magentic-UI logs and LLM inference."
)

# Permitir CORS se necessário
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# 4) Modelos Pydantic para request/response
# ---------------------------------------------------
class ExplainRequest(BaseModel):
    decision_id: str = Field(..., example="123")
    question: str    = Field(..., example="Por que venderam a ação X?")

class ExplainResponse(BaseModel):
    answer: str
    references: List[str]

# ---------------------------------------------------
# 5) Helper functions
# ---------------------------------------------------
async def load_magentic_logs(decision_id: str) -> Dict[str, Any]:
    """
    Busca logs e contexto do Magentic-UI via seu endpoint MCP.
    Raises HTTPException em caso de falha.
    """
    url = f"{MAGENTIC_UI_URL}/api/logs/{decision_id}"
    headers = {"Authorization": f"Bearer {MAGENTIC_UI_TOKEN}"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        logger.error(f"Failed to load logs for {decision_id}: {resp.status_code} {resp.text}")
        raise HTTPException(status_code=502, detail="Error fetching logs from Magentic-UI")
    data = resp.json()
    logger.info(f"Loaded {len(data.get('entries', []))} log entries for decision {decision_id}")
    return data

async def call_llm_explain(logs: Dict[str, Any], question: str) -> (str, List[str]):
    """
    Chama o LLM configurado para gerar explicação em linguagem natural.
    Retorna (answer, references).
    """
    payload = {
        "model": "anthropic/claude-3.7-sonnet",
        "messages": [
            {"role": "system", "content": "Você é um coordenador de sistema financeiro multiagente."},
            {"role": "user", "content": f"Logs: {logs}\n\nPergunta: {question}"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(LLM_API_URL, json=payload, headers=headers)
    if resp.status_code != 200:
        logger.error(f"LLM call failed: {resp.status_code} {resp.text}")
        raise HTTPException(status_code=502, detail="Error during LLM inference")
    result = resp.json()
    # Exemplo de parsing, ajuste conforme a API real
    answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    references = [f"msg_{i}" for i, _ in enumerate(logs.get("entries", []), 1)]
    logger.info(f"Generated explanation of length {len(answer)} chars with {len(references)} refs")
    return answer, references

# ---------------------------------------------------
# 6) Endpoint principal
# ---------------------------------------------------
@app.post("/explain", response_model=ExplainResponse, status_code=200)
async def explain(req: ExplainRequest):
    """
    Gera uma explicação natural-language para a decisão especificada.
    1) Carrega logs do Magentic-UI
    2) Chama LLM para gerar explicação
    3) Retorna answer e referências
    """
    logger.info(f"Received explain request: decision_id={req.decision_id}")
    logs = await load_magentic_logs(req.decision_id)
    answer, references = await call_llm_explain(logs, req.question)
    return ExplainResponse(answer=answer, references=references)

# ---------------------------------------------------
# 7) Eventos de startup/shutdown
# ---------------------------------------------------
@app.on_event("startup")
async def on_startup():
    logger.info("NLWeb Explain Service starting up...")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("NLWeb Explain Service shutting down...")

# ---------------------------------------------------
# 8) Execução direta
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nlweb.nlweb_endpoints:app", host="0.0.0.0", port=8000, reload=True)
