"""
EzioBaseExpert - Base class for all experts in the EzioFilhoUnified system
Handles loading models, tokenizer, generation, and output saving
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

class EzioBaseExpert:
    def __init__(self, expert_type: str, config_path: Optional[Path] = None):
        """
        Initialize the base expert

        Args:
            expert_type: Identifier for the expert (e.g., 'sentiment')
            config_path: Optional path to the JSON config file
        """
        self.expert_type = expert_type
        self.logger = logging.getLogger(f"EzioExpert::{expert_type}")

        # Load model configuration
        self.config_path = config_path or (Path(__file__).resolve().parents[1] / "models_config.json")
        self._load_config()

        # Load model + tokenizer
        self._load_model()

    def _load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)

            if self.expert_type not in configs:
                raise ValueError(f"Expert type '{self.expert_type}' not found in config")

            self.config = configs[self.expert_type]
            self.logger.info(f"✅ Config loaded for expert: {self.expert_type}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {e}")
            raise

    def _load_model(self):
        try:
            model_path = self.config["path"]
            quant = self.config.get("quantization")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.logger.info(f"🧠 Loading model from {model_path} on {self.device}")

            if quant == "4bit" or quant == "8bit":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_4bit=(quant == "4bit"), load_in_8bit=(quant == "8bit"))
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=bnb_config
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.logger.info(f"✅ Model and tokenizer loaded: {model_path}")

        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            raise

    def analyze(self, prompt: str, system_message: Optional[str] = None, max_tokens: int = 512) -> str:
        """
        Generate a response from the model

        Args:
            prompt: The user prompt or query
            system_message: Optional system prompt for instruction tuning
            max_tokens: Max new tokens to generate

        Returns:
            Model-generated string
        """
        try:
            final_prompt = f"{system_message}\n\nUser: {prompt}\n\nAssistant:" if system_message else prompt
            input_ids = self.tokenizer.encode(final_prompt, return_tensors="pt").to(self.device)

            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=self.config.get("temperature", 0.7),
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            self.logger.error(f"❌ Inference failed: {e}")
            return f"ERROR: {e}"

    def save_output(self, prompt: str, output: str, output_path: Optional[Path] = None) -> Path:
        """
        Save prompt and response to a JSON file
        """
        output_path = output_path or Path(__file__).resolve().parents[1] / f"outputs/{self.expert_type}_output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "expert": self.expert_type,
            "model": self.config.get("name", self.expert_type),
            "prompt": prompt,
            "output": output
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 Output saved to {output_path}")
        return output_path
