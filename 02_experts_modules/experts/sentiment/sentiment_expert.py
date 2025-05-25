"""
SentimentExpert - Financial Market Sentiment Analysis Expert
Version: 2.0 | Integrated with EzioBaseExpert Framework
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SentimentExpert")

# Resolve paths and import EzioBaseExpert dynamically
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

try:
    from experts.base_expert import EzioBaseExpert
except ImportError as e:
    logger.error(f"❌ Failed to import EzioBaseExpert: {e}")
    raise

# Default system prompt for LLM sentiment evaluation
SYSTEM_PROMPT = """You are a financial sentiment analysis expert.
Analyze the given text and evaluate the sentiment from a financial/market perspective.
Your analysis must include:
1. Overall sentiment (strongly negative, negative, neutral, positive, or strongly positive)
2. Key sentiment drivers
3. Market implications
4. Confidence level in your assessment

Focus strictly on financial market implications, not general sentiment.
"""

class SentimentExpert(EzioBaseExpert):
    """
    Expert class for analyzing financial sentiment based on EzioBaseExpert foundation.
    Provides fast, structured analysis of market-related news and data.
    """

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__(
            expert_type="sentiment",
            config_path=config_path,
            system_message=SYSTEM_PROMPT
        )
        self.capabilities = ["sentiment_analysis", "market_sentiment"]
        self.priority = 6
        self.timeout_seconds = 8.0

    def process_query(self, query: str, context: Dict[str, Any]) -> Any:
        """
        Process a sentiment query. Required method for QuantumMoE integration.

        Args:
            query: The prompt or user input
            context: Optional context for temperature, max_tokens, etc.

        Returns:
            Tuple of result dictionary and confidence float
        """
        result = self.analyze(
            text=query,
            max_tokens=context.get("max_tokens", 512),
            context=context.get("context"),
            temperature=context.get("temperature", 0.2)
        )

        confidence = 0.8 if result.get("status") == "success" else 0.0
        return result, confidence

    def save_output(self, input_text: str, output: Dict[str, Any], output_path: Path) -> None:
        """
        Persist output of sentiment analysis to disk.
        
        Args:
            input_text: Original query
            output: Dictionary result from analysis
            output_path: Destination file path
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Input:\n{input_text}\n\nOutput:\n{output}")
            logger.info(f"✅ Output saved to {output_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save output: {e}")


# CLI debug mode
if __name__ == "__main__":
    logger.info("🧪 Running standalone test for SentimentExpert...")
    expert = SentimentExpert()
    test_query = (
        "Apple announces record earnings for Q1 2025, beating analyst expectations by 12%. "
        "iPhone sales grew 15% year-over-year."
    )
    context = {"max_tokens": 256}
    result, confidence = expert.process_query(test_query, context)
    print("\n--- RESULTADO ---")
    print(result["response"] if result.get("status") == "success" else result.get("error"))
