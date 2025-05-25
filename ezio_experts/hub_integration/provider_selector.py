# HuggingFace Provider Selector
# Created: 2025-05-24
# Author: marcobarreto007

import os
from typing import Dict, List, Optional
from enum import Enum


class InferenceProvider(Enum):
    """Available inference providers on HF Hub"""
    HF_INFERENCE_API = "hf_api"
    HF_INFERENCE_ENDPOINTS = "hf_endpoints"
    AWS_SAGEMAKER = "aws_sagemaker"


class ProviderSelector:
    def __init__(self):
        self.providers = {}

    def recommend_provider(self, usage):
        print(f"Analyzing usage: {usage}")
        return InferenceProvider.HF_INFERENCE_API


if __name__ == "__main__":
    selector = ProviderSelector()
    result = selector.recommend_provider(1000)
    print(f"Recommended provider: {result}")
