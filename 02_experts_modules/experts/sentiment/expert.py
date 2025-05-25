"""
Sentiment Expert - Specialized in financial sentiment analysis with enhanced auditing
"""
import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Setup enhanced logging for audit mode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SentimentExpert")
audit_logger = logging.getLogger("SentimentExpert.Audit")
audit_logger.setLevel(logging.DEBUG)

# Ensure both base expert implementations are importable
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))

# Try different import paths for compatibility
try:
    # First attempt from experts directory (original location)
    from experts.base_expert import EzioBaseExpert
    logger.info("Successfully imported EzioBaseExpert from experts.base_expert")
except ImportError:
    try:
        # Second attempt from core directory (new location)
        from core.ezio_base_expert import EzioBaseExpert
        logger.info("Successfully imported EzioBaseExpert from core.ezio_base_expert")
    except ImportError as e:
        logger.error(f"Critical: Failed to import EzioBaseExpert from any location: {e}")
        logger.error(f"Python path: {sys.path}")
        logger.error(f"Current directory: {os.getcwd()}")
        raise

# System prompt used for sentiment analysis
SYSTEM_PROMPT = """You are a financial sentiment analysis expert.
Analyze the given text and evaluate the sentiment from a financial/market perspective.
Your analysis should be thorough but concise, and should include:
1. Overall sentiment (strongly negative, negative, neutral, positive, or strongly positive)
2. Key sentiment drivers from the text
3. Potential market implications
4. Confidence level in your assessment
Always focus on financial market implications, not general sentiment."""

class SentimentExpert(EzioBaseExpert):
    """
    SentimentExpert class that uses EzioBaseExpert for financial sentiment analysis.
    Enhanced with audit capabilities for tracking and validation.
    """
    def __init__(self, config_path: Optional[Path] = None, audit_mode: bool = True):
        """
        Initialize the sentiment expert
        
        Args:
            config_path: Optional path to configuration file
            audit_mode: Enable detailed audit logging and validation
        """
        super().__init__(
            expert_type="sentiment",
            config_path=config_path,
            system_message=SYSTEM_PROMPT
        )
        self.capabilities = ["sentiment_analysis", "market_sentiment", "financial_analysis"]
        self.priority = 6  # Moderate priority
        self.timeout_seconds = 8.0  # Fast response expected
        self.audit_mode = audit_mode
        self.audit_log = []
        
        # Log initialization in audit mode
        if self.audit_mode:
            audit_logger.info(f"SentimentExpert initialized in AUDIT MODE with capabilities: {self.capabilities}")
            audit_logger.info(f"Using model: {self.model_name if hasattr(self, 'model_name') else 'Unknown'}")
    
    def process_query(self, query: str, context: Dict[str, Any]) -> Tuple[Any, float]:
        """
        Main interface used by QuantumMoECore. This is required.
        
        Args:
            query: The user prompt or question
            context: Additional metadata or hints (optional)
        
        Returns:
            A tuple: (result_dict, confidence_score)
        """
        # Start audit record if in audit mode
        audit_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "context": context,
            "process_start_time": time.time()
        } if self.audit_mode else None
        
        # Log the incoming query
        logger.info(f"Processing query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Extract parameters from context with defaults
        max_tokens = context.get("max_tokens", 512)
        temp = context.get("temperature", 0.2)
        additional_context = context.get("context", None)
        
        # Process the query
        try:
            result = self.analyze(
                text=query,
                max_tokens=max_tokens,
                context=additional_context,
                temperature=temp
            )
            
            # Calculate confidence based on result status
            if result["status"] == "success":
                # Simple heuristic: higher confidence for lower temperatures
                confidence = min(0.9, 1.0 - temp)
                logger.info(f"Analysis successful, confidence: {confidence:.2f}")
                
                # Additional log in audit mode
                if self.audit_mode:
                    audit_logger.debug(f"Generated response (tokens: {result.get('tokens_generated', 'unknown')})")
            else:
                confidence = 0.0
                logger.warning(f"Analysis failed: {result.get('error', 'Unknown error')}")
            
            # Complete audit record if in audit mode
            if self.audit_mode:
                audit_record.update({
                    "process_end_time": time.time(),
                    "duration": time.time() - audit_record["process_start_time"],
                    "result_status": result["status"],
                    "confidence": confidence
                })
                self.audit_log.append(audit_record)
                
                # Save audit log periodically
                if len(self.audit_log) % 10 == 0:
                    self._save_audit_log()
            
            return result, confidence
            
        except Exception as e:
            logger.error(f"Exception during sentiment analysis: {e}", exc_info=True)
            
            # Complete audit record with error if in audit mode
            if self.audit_mode:
                audit_record.update({
                    "process_end_time": time.time(),
                    "duration": time.time() - audit_record["process_start_time"],
                    "error": str(e),
                    "result_status": "error",
                    "confidence": 0.0
                })
                self.audit_log.append(audit_record)
            
            return {"status": "error", "error": str(e)}, 0.0
    
    def save_output(self, input_text: str, output: Dict[str, Any], output_path: Path) -> None:
        """
        Save analysis result to a file.
        
        Args:
            input_text: The original input text
            output: The result dictionary
            output_path: Path to output file
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format the output
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Input:\n{input_text}\n\n")
                f.write(f"Output:\n{output}")
                
                # Add audit info
                if self.audit_mode:
                    f.write("\n\nAudit Info:\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Model: {self.model_name if hasattr(self, 'model_name') else 'Unknown'}\n")
            
            logger.info(f"Output saved to: {output_path}")
        except Exception as e:
            logger.warning(f"Could not save output to file: {e}")
    
    def _save_audit_log(self) -> None:
        """Save the current audit log to disk"""
        if not self.audit_mode or not self.audit_log:
            return
            
        try:
            audit_dir = Path("./audit_logs").resolve()
            audit_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audit_file = audit_dir / f"sentiment_expert_audit_{timestamp}.json"
            
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(self.audit_log, f, indent=2)
                
            audit_logger.info(f"Audit log saved to: {audit_file}")
        except Exception as e:
            audit_logger.error(f"Failed to save audit log: {e}")

# Debug mode
if __name__ == "__main__":
    logger.info("🔍 Starting standalone test for SentimentExpert in AUDIT MODE...")
    expert = SentimentExpert(audit_mode=True)
    test_query = (
        "Apple announces record earnings for Q1 2025, beating analyst expectations by 12%. "
        "iPhone sales grew 15% year-over-year."
    )
    context = {"max_tokens": 256}
    result, score = expert.process_query(test_query, context)
    print("\n--- RESULTADO ---")
    print(f"Confidence Score: {score:.2f}")
    print(result["response"] if result["status"] == "success" else result["error"])
    
    # Display audit summary
    print("\n--- AUDIT SUMMARY ---")
    print(f"Queries processed: {len(expert.audit_log)}")
    expert._save_audit_log()