# Caminho: C:\Users\anapa\SuperIA\EzioFilhoUnified\main_autogen.py

"""
AutoGen System with Local Model Router - Robust v2
+ Diagnóstico avançado, logging aprimorado e inicialização universal.
"""

import os, sys, json, time, argparse, logging
from pathlib import Path

# 1. Setup Logging Universal e Robustez
def setup_logging(debug=False, log_file=None):
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Remove duplicatas
    level = logging.DEBUG if debug else logging.INFO
    fmt = '%(asctime)s [%(levelname)s] %(name)s – %(message)s'
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt))
        root_logger.addHandler(h)
    root_logger.setLevel(level)

logger = logging.getLogger("main_autogen")

# 2. Helpers universais
PROJECT_ROOT = Path(__file__).parent.absolute()
def require(cond, msg): 
    if not cond:
        logger.error(msg)
        raise RuntimeError(msg)

# 3. Checagem de dependências crítica
def check_dependencies():
    try:
        import autogen
        logger.info("AutoGen OK")
        return True
    except ImportError:
        logger.error("Faltando pyautogen! Execute: pip install pyautogen")
        return False

# 4. Diagnóstico de estrutura
def diagnose_project():
    core = PROJECT_ROOT / "core"
    core_init = core / "__init__.py"
    logger.info(f"ROOT: {PROJECT_ROOT}")
    if not core.exists():
        logger.error(f"core/ não encontrado!")
        return False
    if not core_init.exists():
        logger.warning(f"__init__.py ausente em core/, criando...")
        core_init.write_text("# Package marker\n")
    core_files = list(core.glob("*.py"))
    logger.info(f"core/: {[f.name for f in core_files]}")
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from core import local_model_wrapper, model_router
        logger.info("Módulos core importados com sucesso")
        return True
    except ImportError as e:
        logger.error(f"Importação falhou: {e}")
        return False

# 5. Construção de roteador com fallback
def build_router(config_path=None):
    from core.model_router import create_model_router
    default_configs = [
        {
            "name": "phi2",
            "path": str(Path.home() / ".cache/models/phi-2.gguf"),
            "model_type": "gguf",
            "capabilities": ["fast", "general"]
        },
        {
            "name": "mistral",
            "path": str(Path.home() / ".cache/models/mistral-7b.gguf"),
            "model_type": "gguf",
            "capabilities": ["precise", "general"]
        }
    ]
    config, default_model = default_configs, "phi2"
    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
                config = data.get("models", default_configs)
                default_model = data.get("default_model", "phi2")
    valid = [c for c in config if Path(c["path"]).exists()]
    require(valid, "Nenhum modelo local encontrado!")
    return create_model_router(model_configs=valid, default_model=default_model)

# 6. Função principal enxuta e robusta
def main():
    parser = argparse.ArgumentParser(description="Sistema AutoGen com roteamento local")
    parser.add_argument("--config", help="Arquivo JSON de config dos modelos")
    parser.add_argument("--prompt", default="Explique o que é IA.", help="Prompt de teste")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-file")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    setup_logging(args.debug, args.log_file)
    if args.self_test:
        logger.info("Auto-teste ON")
        ok = check_dependencies() and diagnose_project()
        logger.info("Self-test OK" if ok else "Self-test FALHOU")
        return 0 if ok else 1
    if not check_dependencies():
        return 1
    if not diagnose_project():
        return 1
    router = build_router(args.config)
    import autogen
    agent = autogen.ConversableAgent(name="Generalist", system_message="Você é um assistente versátil.")
    user = autogen.UserProxyAgent(name="User", is_human=False, human_input_mode="NEVER")
    logger.info(f"Prompt: {args.prompt}")
    user.initiate_chat(agent, message=args.prompt)
    return 0

if __name__ == "__main__":
    sys.exit(main())
