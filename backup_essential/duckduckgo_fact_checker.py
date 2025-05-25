#!/usr/bin/env python3
# (venv_ezio) C:\Users\anapa\SuperIA\EzioFilhoUnified\ezio_experts\fact_checker\duckduckgo_fact_checker.py
"""
DuckDuckGo Fact-Checker Expert
------------------------------
Queries DuckDuckGo, extracts the first N snippets and checks if the model's answer
appears (fuzzy match). Returns confidence ∈ [0,1] + best evidence link.

Dependencies (add to requirements_working.txt):
    duckduckgo_search>=4.0.8
    rapidfuzz>=3.6.1
"""

from typing import Tuple
from rapidfuzz import fuzz, process
from duckduckgo_search import DDGS

class DuckDuckGoFactChecker:
    def __init__(self, max_snippets: int = 8):
        self.max_snippets = max_snippets

    def verify(self, question: str, answer: str) -> Tuple[float, str]:
        """Return (confidence, best_url)."""
        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=self.max_snippets))
        if not results:
            return 0.0, ""

        # Combine snippet & title for matching
        candidates = [(r["href"], f"{r['title']} {r['body']}") for r in results]
        best_url, best_score = "", 0
        for url, text in candidates:
            score = fuzz.partial_ratio(answer.lower(), text.lower())
            if score > best_score:
                best_score, best_url = score, url

        confidence = best_score / 100
        return confidence, best_url
