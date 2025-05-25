"""
brain_core.py - Central Brain for EzioFilho_LLMGraph_Alpha
"""
import os, sys, json, time, logging, requests, traceback, threading
import queue, signal, tempfile, psutil, re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field

# Configurar importações
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
for path in [ROOT_DIR, CURRENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Importar ClaudeSyncBridge com fallbacks
try:
    from claude_sync_bridge import ClaudeSyncBridge, logger
except ImportError:
    try:
        from experts.claude_sync.claude_sync_bridge import ClaudeSyncBridge, logger
    except ImportError:
        try:
            from .claude_sync_bridge import ClaudeSyncBridge, logger
        except ImportError:
            raise ImportError("Não foi possível importar ClaudeSyncBridge")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.FileHandler("brain_core.log"), logging.StreamHandler(sys.stdout)]
)
brain_logger = logging.getLogger("BrainCore")

# Verificar e importar dependências
def check_dependencies():
    missing = []
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
    
    if missing:
        brain_logger.warning(f"Dependências faltando: {', '.join(missing)}")
        try:
            import subprocess
            for dep in missing:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            brain_logger.info("Dependências instaladas.")
        except Exception as e:
            brain_logger.error(f"Erro ao instalar dependências: {e}")

check_dependencies()

@dataclass
class ExpertOutput:
    """Saída de especialista."""
    id: str
    expert: str
    timestamp: str
    priority: int = 0
    content: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertOutput':
        for field in ['expert', 'content']:
            if field not in data:
                raise ValueError(f"Campo obrigatório ausente: {field}")
        
        if 'id' not in data:
            data['id'] = f"{data['expert']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        return cls(**data)

@dataclass
class Decision:
    """Estrutura de decisão."""
    id: str
    timestamp: str
    context_id: str
    decision: Dict[str, Any]
    confidence: float
    reasoning: str
    source: str
    expert_outputs: List[Dict[str, Any]]
    requires_feedback: bool = False
    coordination_text: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Decision':
        for field in ['decision', 'confidence', 'reasoning', 'source', 'expert_outputs']:
            if field not in data:
                raise ValueError(f"Campo obrigatório ausente: {field}")
        
        if 'id' not in data:
            data['id'] = f"decision_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        if 'context_id' not in data:
            data['context_id'] = "default"
            
        # Validar confidence
        if not isinstance(data['confidence'], (int, float)):
            try:
                data['confidence'] = float(data['confidence'])
            except:
                data['confidence'] = 0.5
        data['confidence'] = max(0.0, min(1.0, data['confidence']))
        
        return cls(**data)

class CircuitBreaker:
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = None
        self.lock = threading.RLock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == "OPEN":
                    if self.last_failure_time and (datetime.now() - self.last_failure_time).total_seconds() > self.reset_timeout:
                        self.state = "HALF-OPEN"
                    else:
                        raise RuntimeError(f"Circuit breaker aberto para {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                with self.lock:
                    if self.state == "HALF-OPEN":
                        self.failures = 0
                        self.state = "CLOSED"
                return result
            except Exception as e:
                with self.lock:
                    self.failures += 1
                    self.last_failure_time = datetime.now()
                    if self.failures >= self.max_failures:
                        self.state = "OPEN"
                raise
        return wrapper

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            last_exc = None
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    attempt += 1
                    if attempt == max_attempts:
                        brain_logger.error(f"Retry falhou para {func.__name__}: {e}")
                        raise last_exc
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

@contextmanager
def timeout(seconds, error_message="Operação expirou por timeout"):
    if sys.platform == 'win32':
        import threading
        result = {"timed_out": False}
        timer = threading.Timer(seconds, lambda: setattr(result, "timed_out", True))
        try:
            timer.start()
            yield
            if result["timed_out"]:
                raise TimeoutError(error_message)
        finally:
            timer.cancel()
    else:
        def signal_handler(signum, frame):
            raise TimeoutError(error_message)
        
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            signal.alarm(seconds)
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

def safe_json_loads(data, default=None):
    if not data:
        return default
    try:
        return json.loads(data)
    except:
        return default

class CentralBrain(ClaudeSyncBridge):
    """Central Brain for EzioFilho_LLMGraph_Alpha."""
    
    def __init__(self, 
                experts_dir="./experts", 
                decisions_dir="./brain_decisions",
                feedback_dir="./brain_feedback",
                state_file="./brain_state.json",
                max_queue_size=1000,
                health_check_interval=300,
                resource_check_interval=60,
                memory_limit_percent=85.0,
                **kwargs):
        try:
            super().__init__(**kwargs)
            brain_logger.info("ClaudeSyncBridge inicializado")
        except Exception as e:
            brain_logger.critical(f"Falha ao inicializar ClaudeSyncBridge: {e}")
            raise RuntimeError(f"Falha na inicialização: {e}") from e
        
        # Configurações
        self.max_queue_size = max_queue_size
        self.health_check_interval = health_check_interval
        self.resource_check_interval = resource_check_interval
        self.memory_limit_percent = memory_limit_percent
        
        # Diretórios
        self.experts_dir = os.path.abspath(experts_dir)
        self.decisions_dir = os.path.abspath(decisions_dir)
        self.feedback_dir = os.path.abspath(feedback_dir)
        self.state_file = os.path.abspath(state_file)
        
        # Criar diretórios
        for directory in [self.experts_dir, self.decisions_dir, self.feedback_dir]:
            try:
                os.makedirs(directory, exist_ok=True)
            except PermissionError:
                alt_dir = os.path.join(tempfile.gettempdir(), os.path.basename(directory))
                os.makedirs(alt_dir, exist_ok=True)
                if directory == self.experts_dir:
                    self.experts_dir = alt_dir
                elif directory == self.decisions_dir:
                    self.decisions_dir = alt_dir
                else:
                    self.feedback_dir = alt_dir
        
        # State
        self.expert_outputs_queue = queue.Queue(maxsize=max_queue_size)
        self.decision_thread = None
        self.health_check_thread = None
        self.resource_check_thread = None
        self.running = False
        self.degraded_mode = False
        self.health_status = {"status": "starting", "details": {}}
        
        # Locks e estruturas de dados
        self.experts_lock = threading.RLock()
        self.experts = {}
        self.expert_outputs = {}
        self.decisions_lock = threading.RLock()
        self.decisions = []
        self.max_decisions_in_memory = 100
        self.api_failures = {"claude": 0, "github": 0, "last_failure_time": None}
        
        # Inicialização
        try:
            self._load_brain_config()
        except Exception as e:
            brain_logger.error(f"Erro ao carregar configuração: {e}")
            self._create_default_config()
        
        self._restore_state()
        self._last_state_save = datetime.now()
        brain_logger.info("Central Brain inicializado")
    
    def _create_default_config(self):
        """Criar configuração padrão."""
        default_config = {
            "decision_threshold": 0.7,
            "expert_weights": {},
            "decision_aggregation": "weighted_average",
            "github_models_api": {
                "endpoint": "https://api.github.com/models/inference/chat/completions",
                "model": "anthropic/claude-3-7-sonnet",
                "token": None,
            },
            "expert_prompt_templates": {
                "coordination": "You are the central brain of EzioFilho_LLMGraph_Alpha.\nAnalyze these expert outputs and provide a coordinated decision:\n\n{expert_outputs}\n\nYour task is to:\n1. Identify areas of agreement and disagreement\n2. Resolve conflicts using reasoning\n3. Make a final decision\n4. Justify your decision\n5. Provide feedback",
                "feedback": "Review the performance of this expert:\n\nExpert: {expert_name}\nRecent outputs: {expert_outputs}\nDecision alignment: {alignment_score}\n\nProvide specific feedback:"
            },
            "api": {
                "max_retries": 3,
                "retry_delay": 2,
                "timeout_seconds": 30
            },
            "single_expert_passthrough": False
        }
        
        try:
            config_file = os.path.join(os.path.dirname(self.config_file), "brain_config.json")
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.brain_config = default_config
        except Exception as e:
            brain_logger.warning(f"Erro ao salvar configuração: {e}")
            self.brain_config = default_config
    
    def _load_brain_config(self):
        """Carregar configuração."""
        config_file = os.path.join(os.path.dirname(self.config_file), "brain_config.json")
        default_config = self._get_default_config()
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Atualizar valores ausentes recursivamente
                def update_missing(target, source):
                    for key, value in source.items():
                        if key not in target:
                            target[key] = value
                        elif isinstance(value, dict) and isinstance(target[key], dict):
                            update_missing(target[key], value)
                
                update_missing(config, default_config)
                self.brain_config = config
            except:
                self.brain_config = default_config
        else:
            try:
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
            except Exception:
                pass
            self.brain_config = default_config
    
    def _get_default_config(self):
        """Retorna configuração padrão."""
        return {
            "decision_threshold": 0.7,
            "expert_weights": {},
            "decision_aggregation": "weighted_average",
            "github_models_api": {
                "endpoint": "https://api.github.com/models/inference/chat/completions",
                "model": "anthropic/claude-3-7-sonnet",
                "token": None,
            },
            "expert_prompt_templates": {
                "coordination": "You are the central brain of EzioFilho_LLMGraph_Alpha.\nAnalyze these expert outputs and provide a coordinated decision:\n\n{expert_outputs}\n\nYour task is to:\n1. Identify areas of agreement and disagreement\n2. Resolve conflicts using reasoning\n3. Make a final decision\n4. Justify your decision\n5. Provide feedback",
                "feedback": "Review the performance of this expert:\n\nExpert: {expert_name}\nRecent outputs: {expert_outputs}\nDecision alignment: {alignment_score}\n\nProvide specific feedback:"
            },
            "api": {
                "max_retries": 3,
                "retry_delay": 2,
                "timeout_seconds": 30
            },
            "single_expert_passthrough": False
        }
    
    def _save_state(self):
        """Salvar estado do cérebro."""
        try:
            with self.decisions_lock:
                recent_decisions = [asdict(Decision.from_dict(d)) for d in self.decisions[-20:]] if self.decisions else []
            
            with self.experts_lock:
                expert_names = list(self.experts.keys())
            
            state = {
                "timestamp": datetime.now().isoformat(),
                "expert_names": expert_names,
                "recent_decisions": recent_decisions,
                "health_status": self.health_status,
                "degraded_mode": self.degraded_mode,
                "api_failures": self.api_failures
            }
            
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(temp_file, self.state_file)
            self._last_state_save = datetime.now()
        except Exception as e:
            brain_logger.error(f"Erro ao salvar estado: {e}")
    
    def _restore_state(self):
        """Restaurar estado salvo."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            if "recent_decisions" in state and isinstance(state["recent_decisions"], list):
                with self.decisions_lock:
                    for d in state["recent_decisions"]:
                        try:
                            self.decisions.append(asdict(Decision.from_dict(d)))
                        except Exception:
                            pass
            
            if "degraded_mode" in state:
                self.degraded_mode = bool(state["degraded_mode"])
            
            if "api_failures" in state and isinstance(state["api_failures"], dict):
                self.api_failures = state["api_failures"]
            
            brain_logger.info(f"Estado restaurado de {self.state_file}")
        except Exception as e:
            brain_logger.error(f"Erro ao restaurar estado: {e}")
    
    def register_expert(self, expert_name: str, expert_instance: Any) -> bool:
        """Registrar um especialista."""
        if not expert_name or not isinstance(expert_name, str) or expert_instance is None:
            return False
        
        with self.experts_lock:
            if expert_name in self.experts:
                return False
            
            self.experts[expert_name] = expert_instance
            self.expert_outputs[expert_name] = []
        
        brain_logger.info(f"Especialista registrado: {expert_name}")
        return True
    
    def receive_expert_output(self, expert_name: str, output: Dict, priority: int = 0) -> str:
        """Receber saída de especialista."""
        if not expert_name or not isinstance(expert_name, str) or not output or not isinstance(output, dict):
            raise ValueError("Parâmetros inválidos")
        
        try:
            expert_output = ExpertOutput.from_dict({
                "expert": expert_name,
                "content": output,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            })
            
            with self.experts_lock:
                if expert_name in self.expert_outputs:
                    self.expert_outputs[expert_name].append(asdict(expert_output))
                    if len(self.expert_outputs[expert_name]) > 10:
                        self.expert_outputs[expert_name] = self.expert_outputs[expert_name][-10:]
            
            if self.expert_outputs_queue.qsize() >= self.max_queue_size:
                if self.degraded_mode and priority <= 0:
                    return expert_output.id
                
                try:
                    to_remove = max(1, self.max_queue_size // 10)
                    for _ in range(to_remove):
                        try:
                            self.expert_outputs_queue.get_nowait()
                        except queue.Empty:
                            break
                except Exception:
                    pass
            
            try:
                self.expert_outputs_queue.put(asdict(expert_output), timeout=2.0)
                return expert_output.id
            except queue.Full:
                self.degraded_mode = True
                self._save_state()
                raise RuntimeError("Fila de saídas cheia, sistema em modo degradado")
        except Exception as e:
            brain_logger.error(f"Erro ao processar saída do especialista: {e}")
            raise
    
    def start(self):
        """Iniciar todos os threads do cérebro."""
        success = self.start_decision_thread()
        self.start_health_check_thread()
        self.start_resource_check_thread()
        return success
    
    def stop(self):
        """Parar todos os threads do cérebro."""
        self._save_state()
        self.stop_decision_thread()
        self.running = False
        
        threads = []
        if self.health_check_thread and self.health_check_thread.is_alive():
            threads.append(self.health_check_thread)
        if self.resource_check_thread and self.resource_check_thread.is_alive():
            threads.append(self.resource_check_thread)
            
        for thread in threads:
            thread.join(timeout=5.0)
            
        return True
    
    def start_decision_thread(self):
        """Iniciar thread de decisão."""
        if self.decision_thread is not None and self.decision_thread.is_alive():
            return False
        
        self.running = True
        self.decision_thread = threading.Thread(
            target=self._decision_loop, 
            daemon=True,
            name="BrainDecisionThread"
        )
        self.decision_thread.start()
        return True
    
    def stop_decision_thread(self):
        """Parar thread de decisão."""
        if not self.running or not self.decision_thread or not self.decision_thread.is_alive():
            return False
        
        self.running = False
        try:
            self.decision_thread.join(timeout=10.0)
        except Exception:
            pass
        return True
    
    def start_health_check_thread(self):
        """Iniciar thread de verificação de saúde."""
        if self.health_check_thread is not None and self.health_check_thread.is_alive():
            return False
            
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, 
            daemon=True,
            name="BrainHealthCheckThread"
        )
        self.health_check_thread.start()
        return True
    
    def start_resource_check_thread(self):
        """Iniciar thread de monitoramento de recursos."""
        if self.resource_check_thread is not None and self.resource_check_thread.is_alive():
            return False
            
        self.resource_check_thread = threading.Thread(
            target=self._resource_check_loop, 
            daemon=True,
            name="BrainResourceCheckThread"
        )
        self.resource_check_thread.start()
        return True
    
    def _decision_loop(self):
        """Thread de processamento de saídas e tomada de decisões."""
        consecutive_errors = 0
        
        while self.running:
            try:
                consecutive_errors = 0
                batch_outputs = []
                
                try:
                    first_output = self.expert_outputs_queue.get(timeout=1.0)
                    batch_outputs.append(first_output)
                    
                    batch_size = min(10, self.expert_outputs_queue.qsize())
                    for _ in range(batch_size):
                        try:
                            batch_outputs.append(self.expert_outputs_queue.get_nowait())
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue
                
                if batch_outputs:
                    with timeout(30):
                        decisions = self._process_expert_batch(batch_outputs)
                        
                        for decision in decisions:
                            self._save_decision(decision)
                            with self.decisions_lock:
                                self.decisions.append(decision)
                                if len(self.decisions) > self.max_decisions_in_memory:
                                    self.decisions = self.decisions[-self.max_decisions_in_memory:]
                        
                        with self.decisions_lock:
                            if len(self.decisions) % 10 == 0 and len(self.decisions) >= 10:
                                self._provide_expert_feedback()
                
                if datetime.now() - self._last_state_save > timedelta(minutes=5):
                    self._save_state()
                
            except Exception as e:
                consecutive_errors += 1
                brain_logger.error(f"Erro no loop de decisão: {e}")
                
                if consecutive_errors >= 3:
                    self.degraded_mode = True
                    self._save_state()
                
                time.sleep(5)
            
            time.sleep(0.1)
    
    def _health_check_loop(self):
        """Loop de verificação de saúde."""
        while self.running:
            try:
                self._run_health_check()
                self._save_state()
            except Exception as e:
                brain_logger.error(f"Erro na verificação de saúde: {e}")
            
            time.sleep(self.health_check_interval)
    
    def _run_health_check(self):
        """Executar verificação de saúde."""
        health_details = {}
        overall_status = "healthy"
        
        # Verificar threads
        if self.decision_thread and not self.decision_thread.is_alive():
            health_details["threads"] = "decision_thread_dead"
            overall_status = "degraded"
        else:
            health_details["threads"] = "ok"
        
        # Verificar fila
        queue_size = self.expert_outputs_queue.qsize()
        health_details["queue_size"] = queue_size
        
        if queue_size >= self.max_queue_size * 0.9:
            health_details["queue"] = "near_capacity"
            overall_status = "degraded"
        else:
            health_details["queue"] = "ok"
        
        # Verificar APIs
        if self.api_failures["claude"] > 5 or self.api_failures["github"] > 5:
            health_details["api"] = "failing"
            overall_status = "degraded"
        else:
            health_details["api"] = "ok"
        
        # Verificar especialistas
        with self.experts_lock:
            health_details["experts_count"] = len(self.experts)
            if len(self.experts) == 0:
                health_details["experts"] = "none_registered"
                overall_status = "degraded"
        
        # Verificar uso de memória
        try:
            process = psutil.Process(os.getpid())
            memory_percent = process.memory_percent()
            health_details["memory_percent"] = round(memory_percent, 2)
            
            if memory_percent > self.memory_limit_percent:
                health_details["memory"] = "critical"
                overall_status = "degraded"
            elif memory_percent > self.memory_limit_percent * 0.9:
                health_details["memory"] = "high"
            else:
                health_details["memory"] = "ok"
        except Exception:
            health_details["memory"] = "error"
        
        # Atualizar status
        self.health_status = {
            "status": overall_status,
            "details": health_details,
            "timestamp": datetime.now().isoformat(),
            "degraded_mode": self.degraded_mode
        }
        
        if overall_status == "healthy" and self.degraded_mode:
            self.degraded_mode = False
        
        return self.health_status
    
    def _resource_check_loop(self):
        """Loop de monitoramento de recursos."""
        while self.running:
            try:
                # Verificar memória
                process = psutil.Process(os.getpid())
                memory_percent = process.memory_percent()
                
                if memory_percent > self.memory_limit_percent:
                    if not self.degraded_mode:
                        self.degraded_mode = True
                        self._save_state()
                    
                    self._reduce_memory_usage()
                
                # Verificar CPU
                cpu_percent = process.cpu_percent(interval=0.5)
                if cpu_percent > 90 and memory_percent > self.memory_limit_percent * 0.8 and not self.degraded_mode:
                    self.degraded_mode = True
                    self._save_state()
                
            except Exception as e:
                brain_logger.error(f"Erro na verificação de recursos: {e}")
            
            time.sleep(self.resource_check_interval)
    
    def _reduce_memory_usage(self):
        """Reduzir uso de memória."""
        try:
            with self.decisions_lock:
                if len(self.decisions) > 20:
                    self.decisions = self.decisions[-20:]
            
            with self.experts_lock:
                for name in self.expert_outputs:
                    if len(self.expert_outputs[name]) > 5:
                        self.expert_outputs[name] = self.expert_outputs[name][-5:]
            
            if self.expert_outputs_queue.qsize() > self.max_queue_size * 0.7:
                # Limpar fila
                temp_queue = queue.Queue(maxsize=self.max_queue_size)
                transfer_count = 0
                discard_count = 0
                
                while not self.expert_outputs_queue.empty():
                    try:
                        item = self.expert_outputs_queue.get_nowait()
                        if item.get("priority", 0) > 0:
                            try:
                                temp_queue.put_nowait(item)
                                transfer_count += 1
                            except queue.Full:
                                discard_count += 1
                        else:
                            discard_count += 1
                    except queue.Empty:
                        break
                
                self.expert_outputs_queue = temp_queue
            
            # Garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            brain_logger.error(f"Erro ao reduzir memória: {e}")
    
    def _process_expert_batch(self, batch_outputs):
        """Processar lote de saídas de especialistas."""
        if not batch_outputs or not isinstance(batch_outputs, list):
            return []
        
        try:
            # Agrupar por contexto
            context_groups = {}
            
            for output in batch_outputs:
                if not isinstance(output, dict):
                    continue
                
                context_id = "default"
                content = output.get("content", {})
                
                if isinstance(content, dict):
                    context_id = content.get("context_id", "default")
                
                if context_id not in context_groups:
                    context_groups[context_id] = []
                
                context_groups[context_id].append(output)
            
            # Processar cada grupo
            decisions = []
            
            for context_id, outputs in context_groups.items():
                try:
                    if (len(outputs) == 1 and 
                        self.brain_config.get("single_expert_passthrough", False) and 
                        not self.degraded_mode):
                        decision = self._create_decision_from_single_expert(outputs[0])
                    else:
                        decision = self._coordinate_experts(outputs, context_id)
                    
                    decisions.append(decision)
                except Exception as e:
                    brain_logger.error(f"Erro ao processar contexto {context_id}: {e}")
                    
                    # Fallback
                    if outputs:
                        try:
                            highest_priority = max(outputs, key=lambda x: x.get("priority", 0))
                            fallback_decision = self._create_decision_from_single_expert(
                                highest_priority, is_fallback=True
                            )
                            decisions.append(fallback_decision)
                        except Exception:
                            pass
            
            return decisions
            
        except Exception as e:
            brain_logger.error(f"Erro no processamento de lote: {e}")
            return []
    
    def _create_decision_from_single_expert(self, expert_output, is_fallback=False):
        """Criar decisão a partir de um único especialista."""
        try:
            expert_name = expert_output.get("expert", "unknown")
            content = expert_output.get("content", {})
            
            if not isinstance(content, dict):
                content = {"raw_output": str(content)}
            
            decision_id = f"decision_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(str(content))%10000}"
            
            decision_data = {
                "id": decision_id,
                "timestamp": datetime.now().isoformat(),
                "context_id": content.get("context_id", "default"),
                "expert_outputs": [expert_output],
                "decision": content.get("decision", {}),
                "confidence": content.get("confidence", 0.5),
                "reasoning": content.get("reasoning", "Direct passthrough from expert"),
                "source": f"single_expert_{expert_name}",
                "requires_feedback": False
            }
            
            if is_fallback:
                decision_data["source"] = f"fallback_{expert_name}"
                decision_data["confidence"] = max(0.1, decision_data["confidence"] * 0.5)
                decision_data["reasoning"] = f"FALLBACK - {decision_data['reasoning']}"
            
            decision = Decision.from_dict(decision_data)
            return asdict(decision)
            
        except Exception as e:
            brain_logger.error(f"Erro na criação de decisão: {e}")
            
            return {
                "id": f"decision_error_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "context_id": "default",
                "expert_outputs": [expert_output],
                "decision": {"error": str(e)},
                "confidence": 0.1,
                "reasoning": f"Error: {e}",
                "source": "error_recovery",
                "requires_feedback": False
            }
    
    @CircuitBreaker(max_failures=3, reset_timeout=300)
    def _coordinate_experts(self, expert_outputs, context_id):
        """Coordenar especialistas para decisão."""
        if not expert_outputs:
            raise ValueError("Lista de saídas vazia")
        
        try:
            formatted_outputs = self._format_expert_outputs(expert_outputs)
            prompt_template = self.brain_config["expert_prompt_templates"]["coordination"]
            prompt = prompt_template.format(expert_outputs=formatted_outputs)
            
            api_config = self.brain_config.get("api", {})
            max_retries = api_config.get("max_retries", 3)
            retry_delay = api_config.get("retry_delay", 2)
            timeout_seconds = api_config.get("timeout_seconds", 30)
            
            if self.brain_config.get("github_models_api", {}).get("token"):
                response = self._call_github_models_api_with_retry(
                    prompt, max_retries, retry_delay, timeout_seconds
                )
            else:
                response = self._call_claude_api_with_retry(
                    prompt, max_retries, retry_delay, timeout_seconds
                )
            
            return self._extract_decision_from_response(response, expert_outputs, context_id)
            
        except Exception as e:
            brain_logger.error(f"Erro ao coordenar especialistas: {e}")
            
            # Incrementar contador de falhas
            self.api_failures["claude"] += 1
            self.api_failures["last_failure_time"] = datetime.now().isoformat()
            
            return {
                "id": f"decision_error_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "context_id": context_id,
                "expert_outputs": expert_outputs,
                "decision": {"error": str(e), "action": "HOLD"},
                "confidence": 0.1,
                "reasoning": f"Error: {e}",
                "source": "error_recovery",
                "requires_feedback": False
            }
    
    def _format_expert_outputs(self, expert_outputs):
        """Formatar saídas para prompt."""
        if not expert_outputs:
            return "No expert outputs provided."
            
        formatted = []
        
        for output in expert_outputs:
            try:
                expert_name = output.get("expert", "unknown")
                content = output.get("content", {})
                
                if isinstance(content, dict):
                    analysis = content.get("analysis", "No analysis")
                    recommendation = content.get("recommendation", "No recommendation")
                    confidence = content.get("confidence", "Not specified")
                    
                    if isinstance(analysis, str) and len(analysis) > 500:
                        analysis = analysis[:497] + "..."
                    if isinstance(recommendation, str) and len(recommendation) > 300:
                        recommendation = recommendation[:297] + "..."
                    
                    formatted.append(
                        f"## Expert: {expert_name}\n\n"
                        f"**Analysis**: {analysis}\n\n"
                        f"**Recommendation**: {recommendation}\n\n"
                        f"**Confidence**: {confidence}\n\n"
                    )
                else:
                    content_str = str(content)
                    if len(content_str) > 800:
                        content_str = content_str[:797] + "..."
                    formatted.append(f"## Expert: {expert_name}\n\n{content_str}\n\n")
            except Exception as e:
                formatted.append(f"## Expert: {output.get('expert', 'unknown')}\n\nError formatting output: {e}\n\n")
        
        return "\n".join(formatted)
    
    @retry(max_attempts=3, delay=2, backoff=2, exceptions=(requests.RequestException, TimeoutError))
    def _call_github_models_api_with_retry(self, prompt, max_retries=3, retry_delay=2, timeout_seconds=30):
        """Chamar API GitHub Models."""
        github_config = self.brain_config.get("github_models_api", {})
        
        if not github_config.get("token"):
            return self._call_claude_api_with_retry(prompt, max_retries, retry_delay, timeout_seconds)
        
        try:
            headers = {
                "Authorization": f"Bearer {github_config['token']}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28"
            }
            
            payload = {
                "model": github_config.get("model", "anthropic/claude-3-7-sonnet"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.2
            }
            
            response = requests.post(
                github_config.get("endpoint", "https://api.github.com/models/inference/chat/completions"),
                headers=headers,
                json=payload,
                timeout=timeout_seconds
            )
            
            if response.status_code == 200:
                self.api_failures["github"] = 0
                return response.json()
            else:
                self.api_failures["github"] += 1
                self.api_failures["last_failure_time"] = datetime.now().isoformat()
                
                if response.status_code >= 500 or self.api_failures["github"] > 3:
                    return self._call_claude_api_with_retry(prompt, max_retries, retry_delay, timeout_seconds)
                
                raise requests.RequestException(f"API error: {response.status_code}")
                
        except Exception as e:
            self.api_failures["github"] += 1
            self.api_failures["last_failure_time"] = datetime.now().isoformat()
            return self._call_claude_api_with_retry(prompt, max_retries, retry_delay, timeout_seconds)
    
    @retry(max_attempts=3, delay=2, backoff=2, exceptions=(requests.RequestException, TimeoutError))
    def _call_claude_api_with_retry(self, prompt, max_retries=3, retry_delay=2, timeout_seconds=30):
        """Chamar API Claude."""
        if not self.api_key:
            return {"error": "No API key provided"}
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.config.get("claude_model", "claude-3-5-sonnet-20240307"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.2
            }
            
            response = requests.post(
                self.config.get("claude_api_endpoint", "https://api.anthropic.com/v1/messages"),
                headers=headers,
                json=payload,
                timeout=timeout_seconds
            )
            
            if response.status_code == 200:
                self.api_failures["claude"] = 0
                return response.json()
            else:
                self.api_failures["claude"] += 1
                self.api_failures["last_failure_time"] = datetime.now().isoformat()
                raise requests.RequestException(f"API error: {response.status_code}")
                
        except Exception as e:
            self.api_failures["claude"] += 1
            self.api_failures["last_failure_time"] = datetime.now().isoformat()
            raise
    
    def _extract_decision_from_response(self, response, expert_outputs, context_id):
        """Extrair decisão da resposta da API."""
        decision_id = f"decision_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if "error" in response:
            coordination_text = "Erro na coordenação"
            confidence = 0.3
        else:
            if "content" in response:
                # API Claude padrão
                coordination_text = response.get("content", [{}])[0].get("text", "")
            elif "choices" in response:
                # API GitHub Models
                coordination_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                coordination_text = "Formato desconhecido"
            
            confidence = 0.7
        
        decision_data = self._parse_decision_from_text(coordination_text)
        
        try:
            decision = Decision.from_dict({
                "id": decision_id,
                "timestamp": datetime.now().isoformat(),
                "context_id": context_id,
                "expert_outputs": expert_outputs,
                "coordination_text": coordination_text,
                "decision": decision_data.get("decision", {}),
                "confidence": decision_data.get("confidence", confidence),
                "reasoning": decision_data.get("reasoning", coordination_text),
                "source": "central_brain_coordination",
                "requires_feedback": True
            })
            
            return asdict(decision)
            
        except Exception as e:
            brain_logger.error(f"Erro na estrutura de decisão: {e}")
            
            return {
                "id": decision_id,
                "timestamp": datetime.now().isoformat(),
                "context_id": context_id,
                "expert_outputs": expert_outputs,
                "coordination_text": coordination_text[:1000],
                "decision": decision_data.get("decision", {}),
                "confidence": 0.3,
                "reasoning": f"Erro: {e}",
                "source": "error_recovery",
                "requires_feedback": False
            }
    
    def _parse_decision_from_text(self, text):
        """Extrair estrutura de decisão do texto."""
        if not text:
            return {"decision": {}, "confidence": 0.5, "reasoning": "No text"}
            
        decision_data = {"decision": {}, "confidence": 0.7, "reasoning": text}
        
        try:
            # Procurar blocos JSON
            json_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            for json_str in json_matches:
                try:
                    parsed = safe_json_loads(json_str, {})
                    if isinstance(parsed, dict) and any(key in parsed for key in ["decision", "recommendation", "action"]):
                        decision_data.update(parsed)
                        return decision_data
                except:
                    continue
            
            # Procurar seções
            decision_match = re.search(
                r'(?:Decision|Recommendation|Action)[:\s]+(.*?)(?:(?:\n\n)|(?:\n\w)|(?:\Z))', 
                text, re.IGNORECASE | re.DOTALL
            )
            
            if decision_match:
                decision_text = decision_match.group(1).strip()
                decision_data["decision"] = {"text": decision_text}
            
            confidence_match = re.search(r'Confidence[:\s]+([\d.]+)(?:%)?', text, re.IGNORECASE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    if 0 <= confidence <= 1:
                        decision_data["confidence"] = confidence
                    elif 0 <= confidence <= 100:
                        decision_data["confidence"] = confidence / 100
                except:
                    pass
            
            reasoning_match = re.search(
                r'(?:Reasoning|Rationale|Justification)[:\s]+(.*?)(?:(?:\n\n)|(?:\n\w)|(?:\Z))', 
                text, re.IGNORECASE | re.DOTALL
            )
            
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                if reasoning:
                    decision_data["reasoning"] = reasoning
            
            # Extração simples
            if not decision_data["decision"] and len(text) > 0:
                first_para = text.split('\n\n')[0].strip()
                if first_para:
                    decision_data["decision"] = {"text": first_para}
                
        except Exception:
            pass
        
        return decision_data
    
    def _save_decision(self, decision):
        """Salvar decisão."""
        try:
            decision_id = decision["id"]
            decision_file = os.path.join(self.decisions_dir, f"{decision_id}.json")
            
            # Simplificar para arquivo
            simplified = decision.copy()
            simplified["expert_outputs"] = [
                {"id": o.get("id"), "expert": o.get("expert"), "timestamp": o.get("timestamp")}
                for o in decision["expert_outputs"]
            ]
            
            if "coordination_text" in simplified and len(simplified["coordination_text"]) > 2000:
                simplified["coordination_text"] = simplified["coordination_text"][:2000] + "..."
            
            os.makedirs(os.path.dirname(decision_file), exist_ok=True)
            temp_file = f"{decision_file}.tmp"
            
            with open(temp_file, 'w') as f:
                json.dump(simplified, f, indent=2)
            os.replace(temp_file, decision_file)
            
        except Exception as e:
            brain_logger.error(f"Erro ao salvar decisão: {e}")
            
            try:
                alt_file = os.path.join(
                    tempfile.gettempdir(), 
                    f"decision_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                )
                with open(alt_file, 'w') as f:
                    json.dump(decision, f, indent=2)
            except:
                pass
    
    def _provide_expert_feedback(self):
        """Fornecer feedback aos especialistas."""
        try:
            with self.decisions_lock:
                if len(self.decisions) < 10:
                    return
            
            expert_performance = self._calculate_expert_performance()
            
            with self.experts_lock:
                for expert_name, performance in expert_performance.items():
                    if expert_name not in self.experts:
                        continue
                    
                    try:
                        feedback = self._generate_expert_feedback(expert_name, performance)
                        
                        feedback_file = os.path.join(
                            self.feedback_dir, 
                            f"feedback_{expert_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                        )
                        
                        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
                        temp_file = f"{feedback_file}.tmp"
                        with open(temp_file, 'w') as f:
                            json.dump(feedback, f, indent=2)
                            
                        os.replace(temp_file, feedback_file)
                        
                        # Entregar feedback
                        expert = self.experts[expert_name]
                        if hasattr(expert, 'receive_feedback') and callable(getattr(expert, 'receive_feedback')):
                            try:
                                with timeout(10):
                                    expert.receive_feedback(feedback)
                            except Exception:
                                pass
                    except Exception as e:
                        brain_logger.error(f"Erro no feedback para {expert_name}: {e}")
            
        except Exception as e:
            brain_logger.error(f"Erro ao fornecer feedback: {e}")
    
    def _calculate_expert_performance(self):
        """Calcular métricas de desempenho dos especialistas."""
        try:
            with self.decisions_lock:
                recent_decisions = self.decisions[-50:] if len(self.decisions) >= 50 else self.decisions
            
            performance = {}
            
            for decision in recent_decisions:
                try:
                    if decision.get("source") == "central_brain_coordination":
                        for output in decision.get("expert_outputs", []):
                            expert_name = output.get("expert")
                            if not expert_name:
                                continue
                                
                            if expert_name not in performance:
                                performance[expert_name] = {
                                    "contribution_count": 0,
                                    "alignment_score": 0,
                                    "total_confidence": 0,
                                    "recent_outputs": []
                                }
                            
                            performance[expert_name]["contribution_count"] += 1
                            
                            # Cálculo simplificado de alinhamento
                            content = output.get("content", {})
                            expert_decision = content.get("decision", {})
                            central_decision = decision.get("decision", {})
                            
                            alignment = 0.5  # Padrão
                            
                            if isinstance(expert_decision, dict) and isinstance(central_decision, dict):
                                if "text" in expert_decision and "text" in central_decision:
                                    expert_text = str(expert_decision["text"]).lower()
                                    central_text = str(central_decision["text"]).lower()
                                    
                                    if expert_text == central_text:
                                        alignment = 1.0
                                    elif expert_text in central_text or central_text in expert_text:
                                        alignment = 0.8
                                
                                elif "action" in expert_decision and "action" in central_decision:
                                    if expert_decision["action"] == central_decision["action"]:
                                        alignment = 0.8
                                        if expert_decision.get("ticker") == central_decision.get("ticker"):
                                            alignment = 1.0
                            
                            # Atualizar pontuação
                            prev_count = performance[expert_name]["contribution_count"] - 1
                            prev_alignment = performance[expert_name]["alignment_score"]
                            
                            if prev_count > 0:
                                new_alignment = (prev_alignment * prev_count + alignment) / (prev_count + 1)
                            else:
                                new_alignment = alignment
                                
                            performance[expert_name]["alignment_score"] = new_alignment
                            
                            # Manter saídas recentes (até 3)
                            if len(performance[expert_name]["recent_outputs"]) >= 3:
                                performance[expert_name]["recent_outputs"].pop(0)
                            
                            performance[expert_name]["recent_outputs"].append(output)
                except Exception:
                    pass
            
            # Calcular médias
            for expert_name, metrics in performance.items():
                count = metrics["contribution_count"]
                if count > 0:
                    metrics["avg_confidence"] = metrics["total_confidence"] / count
                else:
                    metrics["avg_confidence"] = 0
            
            return performance
            
        except Exception as e:
            brain_logger.error(f"Erro no cálculo de desempenho: {e}")
            return {}
    
    @retry(max_attempts=2, delay=1, backoff=2)
    def _generate_expert_feedback(self, expert_name, performance):
        """Gerar feedback para um especialista."""
        prompt_template = self.brain_config["expert_prompt_templates"]["feedback"]
        formatted_outputs = self._format_expert_outputs(performance.get("recent_outputs", []))
        
        prompt = prompt_template.format(
            expert_name=expert_name,
            expert_outputs=formatted_outputs,
            alignment_score=f"{performance.get('alignment_score', 0):.2f}"
        )
        
        api_config = self.brain_config.get("api", {})
        max_retries = api_config.get("max_retries", 3)
        retry_delay = api_config.get("retry_delay", 2)
        timeout_seconds = api_config.get("timeout_seconds", 30)
        
        if self.brain_config.get("github_models_api", {}).get("token"):
            response = self._call_github_models_api_with_retry(
                prompt, max_retries, retry_delay, timeout_seconds
            )
        else:
            response = self._call_claude_api_with_retry(
                prompt, max_retries, retry_delay, timeout_seconds
            )
        
        # Extrair texto
        if "error" in response:
            feedback_text = f"Erro ao gerar feedback: {response['error']}"
        else:
            if "content" in response:
                feedback_text = response.get("content", [{}])[0].get("text", "")
            elif "choices" in response:
                feedback_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                feedback_text = "Formato desconhecido"
        
        if len(feedback_text) > 5000:
            feedback_text = feedback_text[:5000] + "..."
        
        return {
            "expert_name": expert_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "contribution_count": performance.get("contribution_count", 0),
                "alignment_score": performance.get("alignment_score", 0),
                "avg_confidence": performance.get("avg_confidence", 0)
            },
            "feedback_text": feedback_text,
            "feedback_id": f"feedback_{expert_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    
    def handle_coordination_request(self, request_data):
        """Processar requisição de coordenação."""
        if not isinstance(request_data, dict):
            return {"status": "error", "error": "Dados inválidos"}
            
        context_id = request_data.get("context_id", f"ctx_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        expert_outputs = request_data.get("expert_outputs", [])
        
        if not expert_outputs:
            return {"status": "error", "error": "Nenhuma saída fornecida"}
        
        try:
            decision = self._coordinate_experts(expert_outputs, context_id)
            
            # Salvar e registrar decisão
            self._save_decision(decision)
            with self.decisions_lock:
                self.decisions.append(decision)
                if len(self.decisions) > self.max_decisions_in_memory:
                    self.decisions = self.decisions[-self.max_decisions_in_memory:]
            
            return {
                "status": "success",
                "decision_id": decision["id"],
                "decision": decision["decision"],
                "confidence": decision["confidence"],
                "reasoning": decision["reasoning"],
                "timestamp": decision["timestamp"]
            }
            
        except Exception as e:
            brain_logger.error(f"Erro na coordenação: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def handle_request(self, request_type, request_data):
        """Processar diferentes tipos de requisições."""
        if not request_type or not isinstance(request_type, str) or not isinstance(request_data, dict):
            return {"status": "error", "error": "Requisição inválida"}
        
        try:
            # Requisições específicas do cérebro
            if request_type == "coordinate_experts":
                return self.handle_coordination_request(request_data)
                
            elif request_type == "register_expert":
                expert_name = request_data.get("expert_name")
                expert_instance = request_data.get("expert_instance")
                
                if not expert_name or not expert_instance:
                    return {"status": "error", "error": "Dados incompletos"}
                    
                success = self.register_expert(expert_name, expert_instance)
                return {"status": "success" if success else "error"}
                
            elif request_type == "expert_output":
                expert_name = request_data.get("expert_name")
                output = request_data.get("output")
                priority = request_data.get("priority", 0)
                
                if not expert_name or output is None:
                    return {"status": "error", "error": "Dados incompletos"}
                    
                try:
                    output_id = self.receive_expert_output(expert_name, output, priority)
                    return {"status": "success", "output_id": output_id}
                except queue.Full:
                    return {"status": "error", "error": "Fila cheia", "queue_size": self.expert_outputs_queue.qsize()}
                    
            elif request_type == "start_brain":
                success = self.start()
                return {"status": "success" if success else "error"}
                
            elif request_type == "stop_brain":
                success = self.stop()
                return {"status": "success" if success else "error"}
                
            elif request_type == "get_decision":
                decision_id = request_data.get("decision_id")
                if not decision_id:
                    return {"status": "error", "error": "ID ausente"}
                    
                # Procurar na memória
                with self.decisions_lock:
                    for decision in self.decisions:
                        if decision["id"] == decision_id:
                            return {"status": "success", "decision": decision}
                    
                # Procurar em disco
                decision_file = os.path.join(self.decisions_dir, f"{decision_id}.json")
                if os.path.exists(decision_file):
                    try:
                        with open(decision_file, 'r') as f:
                            decision = json.load(f)
                        return {"status": "success", "decision": decision}
                    except Exception as e:
                        return {"status": "error", "error": f"Erro ao carregar decisão: {e}"}
                        
                return {"status": "error", "error": "Decisão não encontrada"}
                
            elif request_type == "health_check":
                health_status = self._run_health_check()
                return {"status": "success", "health": health_status, "timestamp": datetime.now().isoformat()}
                
            elif request_type == "reset_degraded_mode":
                previous_mode = self.degraded_mode
                self.degraded_mode = False
                self._save_state()
                return {"status": "success", "previous_mode": previous_mode, "current_mode": self.degraded_mode}
                
            # Delegar para a classe base
            return super().handle_request(request_type, request_data)
            
        except Exception as e:
            brain_logger.error(f"Erro na requisição {request_type}: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "request_type": request_type,
                "timestamp": datetime.now().isoformat()
            }


def main():
    """Main entry point for the Central Brain."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Central Brain for EzioFilho_LLMGraph_Alpha")
    parser.add_argument("--reports-dir", type=str, help="Reports directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--api-key", type=str, help="Claude API key")
    parser.add_argument("--experts-dir", type=str, help="Experts directory")
    parser.add_argument("--decisions-dir", type=str, help="Decisions directory")
    parser.add_argument("--feedback-dir", type=str, help="Feedback directory")
    parser.add_argument("--state-file", type=str, default="./brain_state.json", help="State file")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--with-brain", action="store_true", help="Run with brain thread")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Configure log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    brain_logger.setLevel(getattr(logging, args.log_level))
    
    # Build kwargs from args
    brain_kwargs = {}
    for key, value in vars(args).items():
        if value is not None and key not in ["continuous", "with_brain", "interval", "log_level"]:
            brain_kwargs[key] = value
    
    try:
        brain = CentralBrain(**brain_kwargs)
        
        # Setup signal handlers (except on Windows)
        if sys.platform != 'win32':
            def signal_handler(signum, frame):
                brain_logger.info(f"Sinal recebido: {signum}")
                brain.stop()
                sys.exit(0)
                
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        
        # Start threads
        if args.with_brain:
            brain.start()
        elif args.continuous:
            brain.start_decision_thread()
        
        # Run
        if args.continuous:
            brain.run_continuously(args.interval)
        else:
            results = brain.run_once()
            print(f"Processados {len(results)} relatórios")
        
        # Stop threads
        if args.with_brain or args.continuous:
            brain.stop()
            
    except Exception as e:
        brain_logger.critical(f"Erro fatal: {e}")
        brain_logger.debug(f"Stacktrace: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()