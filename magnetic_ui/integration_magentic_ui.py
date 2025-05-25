"""
integration_magentic_ui.py

Micro-serviço FastAPI que:
1) Gerencia o registro de agentes financeiros (Orchestrator, WebSurfer)
2) Serve logs para decisões do EzioFilho via endpoint MCP
3) Expõe hooks HTTP para disparar workflows do EzioFilho
"""
import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------------------------------------------
# 1) Carregar configuração de ambiente
# ---------------------------------------------------
load_dotenv()  # Lê .env na raiz do projeto

MAGENTIC_UI_TOKEN = os.getenv("MAGENTIC_UI_TOKEN", "")
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ---------------------------------------------------
# 2) Configurar logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("magentic_ui")

# ---------------------------------------------------
# 3) Inicializar FastAPI
# ---------------------------------------------------
app = FastAPI(
    title="Magentic-UI Integration Service",
    version="1.0.0",
    description="Serviço para registro de agentes, logs MCP e triggers de workflows EzioFilho."
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# 4) Modelos Pydantic
# ---------------------------------------------------
class LogEntry(BaseModel):
    id: str
    timestamp: str
    message: str

class AgentConfig(BaseModel):
    name: str = Field(..., example="Orchestrator")
    version: str = Field(..., example="1.0.0")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowTrigger(BaseModel):
    workflow_id: str = Field(..., example="trade_execution")
    parameters: Dict[str, Any] = Field(default_factory=dict)

# Armazenamento em memória (exemplo)
registered_agents: Dict[str, AgentConfig] = {}
logs_store: Dict[str, List[LogEntry]] = {}

# ---------------------------------------------------
# 5) Endpoints
# ---------------------------------------------------
@app.get("/api/logs/{decision_id}", response_model=List[LogEntry])
async def get_logs(decision_id: str):
    """
    Retorna a lista de logs para uma decisão específica.
    """
    entries = logs_store.get(decision_id)
    if entries is None:
        logger.warning(f"No logs found for decision_id={decision_id}")
        raise HTTPException(status_code=404, detail="Logs not found")
    logger.info(f"Returning {len(entries)} logs for decision_id={decision_id}")
    return entries

@app.post("/api/agents", status_code=201)
async def register_agent(config: AgentConfig):
    """
    Registra um agente financeiro para EzioFilho.
    """
    if config.name in registered_agents:
        logger.warning(f"Agent '{config.name}' already registered")
        raise HTTPException(status_code=409, detail="Agent already registered")
    registered_agents[config.name] = config
    logger.info(f"Registered agent: {config.name} v{config.version}")
    return {"message": "Agent registered", "agent": config}

@app.post("/api/trigger-workflow", status_code=202)
async def trigger_workflow(trigger: WorkflowTrigger):
    """
    Hook para disparar um workflow do EzioFilho.
    """
    # Aqui, dispare a lógica de workflow real, ex.: colocar no Kafka ou chamar API do EzioFilho
    logger.info(f"Triggering workflow '{trigger.workflow_id}' with params {trigger.parameters}")
    # Simula adição de log
    decision_id = trigger.parameters.get("decision_id", "_unknown")
    entry = LogEntry(
        id=str(len(logs_store.get(decision_id, [])) + 1),
        timestamp="2025-05-20T12:00:00Z",
        message=f"Workflow {trigger.workflow_id} triggered"
    )
    logs_store.setdefault(decision_id, []).append(entry)
    return {"message": "Workflow triggered", "log_entry": entry}

# ---------------------------------------------------
# 6) Eventos de ciclo de vida
# ---------------------------------------------------
@app.on_event("startup")
async def on_startup():
    logger.info("Magentic-UI Integration Service starting up...")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Magentic-UI Integration Service shutting down...")

# ---------------------------------------------------
# 7) Execução direta
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "magentic_ui.integration_magentic_ui:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )
