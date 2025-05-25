import os
import re
import logging
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path

from core.local_model_wrapper import (
    create_model_wrapper,
    LocalModelWrapper  # Corrigido para o nome real
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelRouter")


class ModelCapability(Enum):
    FAST = "fast"
    PRECISE = "precise"
    CODE = "code"
    CREATIVE = "creative"
    SUMMARY = "summary"
    TRANSLATION = "translation"
    GENERAL = "general"


class ModelDefinition:
    def __init__(
        self,
        name: str,
        path: str,
        model_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[ModelCapability]] = None,
        min_prompt_tokens: int = 0,
        max_prompt_tokens: int = 8192,
        supported_languages: Optional[Set[str]] = None,
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.path = str(Path(path).expanduser())
        self.model_type = model_type
        self.params = params or {}
        self.capabilities = capabilities or [ModelCapability.GENERAL]
        self.min_prompt_tokens = min_prompt_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.supported_languages = supported_languages or {"en"}
        self.tags = tags or []
        self._model_instance: Optional[LocalModelWrapper] = None
        self._last_used = 0

    def get_model(self) -> LocalModelWrapper:
        if not self._model_instance:
            logger.info(f"Carregando modelo: {self.name}")
            self._model_instance = create_model_wrapper(
                self.path, model_type=self.model_type, **self.params
            )
        self._last_used = time.time()
        return self._model_instance

    def has_capability(self, cap: ModelCapability) -> bool:
        return cap in self.capabilities

    def supports_language(self, lang: str) -> bool:
        return lang in self.supported_languages

    def unload(self):
        if self._model_instance:
            logger.info(f"Descarregando modelo: {self.name}")
            self._model_instance.unload()
            self._model_instance = None


class ModelRouter:
    def __init__(self, max_loaded_models: int = 3):
        self.models: Dict[str, ModelDefinition] = {}
        self.default_model: Optional[str] = None
        self.max_loaded_models = max_loaded_models

    def register_model(self, model_def: ModelDefinition):
        self.models[model_def.name] = model_def
        logger.info(f"Modelo registrado: {model_def.name}")
        if not self.default_model:
            self.default_model = model_def.name

    def set_default_model(self, name: str):
        if name in self.models:
            self.default_model = name

    def _estimate_tokens(self, text: str) -> int:
        return int(len(re.findall(r'\b\w+\b', text)) * 1.33)

    def _detect_capabilities(self, prompt: str) -> List[ModelCapability]:
        prompt = prompt.lower()
        result = []
        if any(word in prompt for word in ["function", "code", "script"]):
            result.append(ModelCapability.CODE)
        if any(word in prompt for word in ["story", "imagine", "write"]):
            result.append(ModelCapability.CREATIVE)
        if not result:
            result.append(ModelCapability.GENERAL)
        return result

    def _score_model(self, model: ModelDefinition, prompt: str, caps: List[ModelCapability]) -> int:
        score = 0
        tokens = self._estimate_tokens(prompt)
        if tokens > model.max_prompt_tokens:
            return -1000
        if tokens >= model.min_prompt_tokens:
            score += 30
        for cap in caps:
            if model.has_capability(cap):
                score += 25
        return score

    def _select_model(self, prompt: str) -> ModelDefinition:
        caps = self._detect_capabilities(prompt)
        scores = {
            name: self._score_model(model, prompt, caps)
            for name, model in self.models.items()
        }
        best = max(scores.items(), key=lambda x: x[1])
        return self.models.get(best[0], self.models[self.default_model])

    def _manage_memory(self):
        loaded = [(name, m._last_used) for name, m in self.models.items() if m._model_instance]
        if len(loaded) <= self.max_loaded_models:
            return
        loaded.sort(key=lambda x: x[1])
        for name, _ in loaded[:-self.max_loaded_models]:
            self.models[name].unload()

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            model = self._select_model(prompt)
            result = model.get_model().generate(prompt, **(context or {}))
            self._manage_memory()
            return result
        except Exception as e:
            logger.error(f"Erro na geração com roteador: {e}")
            if self.default_model:
                try:
                    return self.models[self.default_model].get_model().generate(prompt)
                except Exception as fallback_e:
                    raise RuntimeError(f"Fallback falhou: {fallback_e}")
            raise


def create_model_router(model_configs: List[Dict[str, Any]], default_model: Optional[str] = None) -> ModelRouter:
    router = ModelRouter()
    for conf in model_configs:
        name = conf.pop("name")
        path = conf.pop("path")
        model_type = conf.pop("model_type", None)
        caps = [ModelCapability(cap.lower()) for cap in conf.pop("capabilities", ["general"])]
        model_def = ModelDefinition(
            name=name,
            path=path,
            model_type=model_type,
            capabilities=caps,
            **conf
        )
        router.register_model(model_def)
    if default_model:
        router.set_default_model(default_model)
    return router
