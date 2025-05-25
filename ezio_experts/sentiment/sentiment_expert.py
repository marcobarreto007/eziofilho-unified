"""
Sentiment Expert - Specialized in financial sentiment analysis
"""
import sys
from pathlib import Path
from typing import Optional

# Import directly from our package
from ezio_experts.ezio_base_expert import EzioBaseExpert

SYSTEM_PROMPT = """You are a financial sentiment analysis expert. 
Analyze the given text and evaluate the sentiment from a financial/market perspective.
Your analysis should be thorough but concise, and should include:
1. Overall sentiment (strongly negative, negative, neutral, positive, or strongly positive)
2. Key sentiment drivers from the text
3. Potential market implications
4. Confidence level in your assessment

Always focus on financial market implications, not general sentiment."""

class SentimentExpert(EzioBaseExpert):
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__("sentiment", config_path)
    
    def analyze_sentiment(self, text: str, output_path: Optional[Path] = None) -> str:
        """Analyze financial sentiment in the given text"""
        output = self.analyze(text, system_message=SYSTEM_PROMPT)
        
        if output_path:
            self.save_output(text, output, output_path)
            
        return output

# Example usage
if __name__ == "__main__":
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python path: {sys.path}")
    
    expert = SentimentExpert()
    result = expert.analyze_sentiment("Apple announces record earnings for Q1 2025, beating analyst expectations by 12%. iPhone sales grew 15% year-over-year.")
    print(result)