"""
claude_sync_bridge.py - Bridge between EzioFinisher and Claude

This script monitors EzioFinisher reports, analyzes them, and automatically
generates prompts for Claude based on report findings.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import re
import glob
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("claude_sync_bridge.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ClaudeSyncBridge")

class ClaudeSyncBridge:
    """
    Bridge between EzioFinisher reports and Claude AI.
    Monitors reports and generates appropriate prompts.
    """
    
    def __init__(self, 
                 reports_dir: str = "./reports/ezio_finisher",
                 output_dir: str = "./claude_prompts",
                 config_file: str = "./config/claude_bridge_config.json",
                 api_key: Optional[str] = None):
        """
        Initialize the Claude Sync Bridge.
        
        Args:
            reports_dir: Directory containing EzioFinisher reports
            output_dir: Directory to save generated Claude prompts
            config_file: Configuration file path
            api_key: Optional API key for direct Claude communication
        """
        self.reports_dir = os.path.abspath(reports_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.config_file = os.path.abspath(config_file)
        self.api_key = api_key
        
        # Ensure directories exist
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Track processed reports
        self.processed_reports = self._load_processed_reports()
        
        logger.info(f"Claude Sync Bridge initialized with reports dir: {self.reports_dir}")
    
    def _load_config(self) -> Dict:
        """Load the bridge configuration."""
        default_config = {
            "prompt_templates": {
                "performance_alert": "Analyze this trading performance alert: {details}\n\nIdentify key issues and recommend actionable steps.",
                "system_status": "Review EzioFilhoUnified system status:\n\n{status}\n\nProvide an assessment and optimization recommendations.",
                "error_report": "Debug these errors from EzioFilhoUnified:\n\n{errors}\n\nSuggest fixes and preventative measures.",
                "strategy_evaluation": "Evaluate this trading strategy performance:\n\n{metrics}\n\nWhat improvements would you suggest?",
                "market_analysis": "Analyze this market data summary:\n\n{market_data}\n\nIdentify notable patterns and trading opportunities."
            },
            "scan_interval_seconds": 300,  # 5 minutes
            "report_types": ["performance", "system", "error", "strategy", "market"],
            "claude_api_endpoint": "https://api.anthropic.com/v1/messages",
            "claude_model": "claude-3-5-sonnet-20240307"
        }
        
        if not os.path.exists(self.config_file):
            # Create default config
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default configuration at {self.config_file}")
            return default_config
            
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_file}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            logger.info("Using default configuration")
            return default_config
    
    def _load_processed_reports(self) -> List[str]:
        """Load list of already processed reports."""
        processed_file = os.path.join(self.output_dir, "processed_reports.json")
        if os.path.exists(processed_file):
            try:
                with open(processed_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Invalid processed reports file. Starting fresh.")
                return []
        return []
    
    def _save_processed_reports(self):
        """Save list of processed reports."""
        processed_file = os.path.join(self.output_dir, "processed_reports.json")
        with open(processed_file, 'w') as f:
            json.dump(self.processed_reports, f, indent=2)
    
    def scan_for_new_reports(self) -> List[str]:
        """
        Scan for new EzioFinisher reports.
        
        Returns:
            List of paths to new reports
        """
        all_reports = []
        
        # Support multiple file formats that EzioFinisher might use
        for ext in ['json', 'csv', 'txt', 'html']:
            pattern = os.path.join(self.reports_dir, f"**/*.{ext}")
            all_reports.extend(glob.glob(pattern, recursive=True))
        
        # Filter out already processed reports
        new_reports = [r for r in all_reports if r not in self.processed_reports]
        
        if new_reports:
            logger.info(f"Found {len(new_reports)} new reports")
        return new_reports
    
    def parse_report(self, report_path: str) -> Dict:
        """
        Parse an EzioFinisher report.
        
        Args:
            report_path: Path to the report file
            
        Returns:
            Parsed report data
        """
        try:
            file_ext = os.path.splitext(report_path)[1].lower()
            
            if file_ext == '.json':
                with open(report_path, 'r') as f:
                    data = json.load(f)
                    
            elif file_ext == '.csv':
                import pandas as pd
                data = pd.read_csv(report_path).to_dict(orient='records')
                
            elif file_ext == '.html':
                from bs4 import BeautifulSoup
                with open(report_path, 'r') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    # Extract data from HTML - simplified for this example
                    data = {
                        "title": soup.title.string if soup.title else "Unknown",
                        "content": soup.get_text(),
                        "timestamp": datetime.now().isoformat()
                    }
                    
            else:  # Assume plain text
                with open(report_path, 'r') as f:
                    content = f.read()
                data = {
                    "content": content,
                    "filename": os.path.basename(report_path),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Add metadata
            data["report_path"] = report_path
            data["report_type"] = self._detect_report_type(data)
            
            logger.info(f"Parsed report: {os.path.basename(report_path)} (Type: {data['report_type']})")
            return data
            
        except Exception as e:
            logger.error(f"Error parsing report {report_path}: {e}")
            return {
                "error": str(e),
                "report_path": report_path,
                "report_type": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def _detect_report_type(self, data: Dict) -> str:
        """
        Detect the type of report based on its content.
        
        Args:
            data: Parsed report data
            
        Returns:
            Report type string
        """
        content = ""
        
        # Extract textual content from different report formats
        if isinstance(data, dict):
            if "content" in data:
                content = data["content"]
            elif "title" in data:
                content = data["title"]
                if "content" in data:
                    content += " " + data["content"]
            else:
                # Convert all values to string and join
                content = " ".join(str(v) for v in data.values() if not isinstance(v, (dict, list)))
                
        # Keywords for each report type
        type_keywords = {
            "performance": ["performance", "return", "profit", "loss", "gain", "pnl", "drawdown"],
            "system": ["system", "status", "health", "memory", "cpu", "uptime", "latency"],
            "error": ["error", "exception", "failure", "crash", "bug", "issue", "warning"],
            "strategy": ["strategy", "algorithm", "backtest", "optimization", "parameter", "model"],
            "market": ["market", "price", "trend", "volatility", "momentum", "volume", "correlation"]
        }
        
        # Count keyword matches for each type
        type_scores = {t: 0 for t in type_keywords}
        
        for t, keywords in type_keywords.items():
            for keyword in keywords:
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content, re.IGNORECASE))
                type_scores[t] += matches
                
        # Return the type with the highest score, or 'unknown' if all scores are 0
        max_score = max(type_scores.values())
        if max_score > 0:
            for t, score in type_scores.items():
                if score == max_score:
                    return t
        
        return "unknown"
    
    def generate_prompt(self, report_data: Dict) -> str:
        """
        Generate a Claude-ready prompt based on report data.
        
        Args:
            report_data: Parsed report data
            
        Returns:
            Formatted prompt for Claude
        """
        report_type = report_data.get("report_type", "unknown")
        
        # Get template for this report type
        template = self.config["prompt_templates"].get(
            f"{report_type}_alert" if report_type in ["performance", "error"] else
            f"{report_type}_status" if report_type == "system" else
            f"{report_type}_evaluation" if report_type == "strategy" else
            f"{report_type}_analysis" if report_type == "market" else
            "general"
        )
        
        # Fallback to general template
        if not template:
            template = "Please analyze this EzioFilhoUnified report:\n\n{details}\n\nProvide insights and recommendations."
        
        # Extract key details based on report type
        if report_type == "performance":
            details = self._extract_performance_details(report_data)
        elif report_type == "system":
            details = self._extract_system_details(report_data)
        elif report_type == "error":
            details = self._extract_error_details(report_data)
        elif report_type == "strategy":
            details = self._extract_strategy_details(report_data)
        elif report_type == "market":
            details = self._extract_market_details(report_data)
        else:
            # For unknown types, include all data
            details = json.dumps(report_data, indent=2)
            
        # Format prompt with details
        prompt = template.format(
            details=details,
            status=details if report_type == "system" else "",
            errors=details if report_type == "error" else "",
            metrics=details if report_type == "strategy" else "",
            market_data=details if report_type == "market" else ""
        )
        
        # Add metadata and instructions
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_file = os.path.basename(report_data.get("report_path", "unknown"))
        
        prompt = f"""
# EzioFilhoUnified Report Analysis Request
- **Report Type**: {report_type.capitalize()}
- **Report File**: {report_file}
- **Generated**: {timestamp}

{prompt}

## Additional Instructions
- Focus on actionable insights
- Identify any concerning patterns
- Suggest specific improvements
- Prioritize recommendations by impact
"""
        
        logger.info(f"Generated prompt for report: {report_file}")
        return prompt
    
    def _extract_performance_details(self, data: Dict) -> str:
        """Extract performance details from report data."""
        details = []
        
        # Try to extract performance metrics
        if isinstance(data, dict):
            # Look for performance-related keys
            performance_keys = ["returns", "profit", "loss", "pnl", "performance", "metrics"]
            
            for key in performance_keys:
                if key in data:
                    if isinstance(data[key], (dict, list)):
                        details.append(f"## {key.capitalize()}")
                        details.append(json.dumps(data[key], indent=2))
                    else:
                        details.append(f"## {key.capitalize()}")
                        details.append(str(data[key]))
            
            # Look for strategy performance
            if "strategies" in data and isinstance(data["strategies"], list):
                details.append("## Strategy Performance")
                for i, strategy in enumerate(data["strategies"]):
                    details.append(f"### Strategy {i+1}")
                    for k, v in strategy.items():
                        details.append(f"- {k}: {v}")
        
        # If no structured data found, use content field or raw data
        if not details:
            if "content" in data:
                return data["content"]
            else:
                return json.dumps(data, indent=2)
                
        return "\n\n".join(details)
    
    def _extract_system_details(self, data: Dict) -> str:
        """Extract system status details from report data."""
        details = []
        
        if isinstance(data, dict):
            # Look for system status keys
            system_keys = ["status", "health", "resources", "memory", "cpu", "uptime"]
            
            for key in system_keys:
                if key in data:
                    if isinstance(data[key], (dict, list)):
                        details.append(f"## {key.capitalize()}")
                        details.append(json.dumps(data[key], indent=2))
                    else:
                        details.append(f"## {key.capitalize()}")
                        details.append(str(data[key]))
                        
            # Look for component status
            if "components" in data and isinstance(data["components"], dict):
                details.append("## Component Status")
                for component, status in data["components"].items():
                    details.append(f"### {component}")
                    if isinstance(status, dict):
                        for k, v in status.items():
                            details.append(f"- {k}: {v}")
                    else:
                        details.append(f"- Status: {status}")
        
        # If no structured data found, use content field or raw data
        if not details:
            if "content" in data:
                return data["content"]
            else:
                return json.dumps(data, indent=2)
                
        return "\n\n".join(details)
    
    def _extract_error_details(self, data: Dict) -> str:
        """Extract error details from report data."""
        details = []
        
        if isinstance(data, dict):
            # Look for error-related keys
            error_keys = ["errors", "exceptions", "warnings", "issues"]
            
            for key in error_keys:
                if key in data:
                    if isinstance(data[key], (dict, list)):
                        details.append(f"## {key.capitalize()}")
                        details.append(json.dumps(data[key], indent=2))
                    else:
                        details.append(f"## {key.capitalize()}")
                        details.append(str(data[key]))
                        
            # Look for traceback
            if "traceback" in data:
                details.append("## Error Traceback")
                details.append("```")
                details.append(data["traceback"])
                details.append("```")
        
        # If no structured data found, use content field or raw data
        if not details:
            if "content" in data:
                return data["content"]
            else:
                return json.dumps(data, indent=2)
                
        return "\n\n".join(details)
    
    def _extract_strategy_details(self, data: Dict) -> str:
        """Extract strategy details from report data."""
        details = []
        
        if isinstance(data, dict):
            # Look for strategy-related keys
            strategy_keys = ["strategy", "parameters", "optimization", "backtest"]
            
            for key in strategy_keys:
                if key in data:
                    if isinstance(data[key], (dict, list)):
                        details.append(f"## {key.capitalize()}")
                        details.append(json.dumps(data[key], indent=2))
                    else:
                        details.append(f"## {key.capitalize()}")
                        details.append(str(data[key]))
                        
            # Look for metrics
            if "metrics" in data:
                details.append("## Performance Metrics")
                metrics = data["metrics"]
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        details.append(f"- {k}: {v}")
                else:
                    details.append(str(metrics))
        
        # If no structured data found, use content field or raw data
        if not details:
            if "content" in data:
                return data["content"]
            else:
                return json.dumps(data, indent=2)
                
        return "\n\n".join(details)
    
    def _extract_market_details(self, data: Dict) -> str:
        """Extract market details from report data."""
        details = []
        
        if isinstance(data, dict):
            # Look for market-related keys
            market_keys = ["market", "prices", "trends", "indicators", "analysis"]
            
            for key in market_keys:
                if key in data:
                    if isinstance(data[key], (dict, list)):
                        details.append(f"## {key.capitalize()}")
                        details.append(json.dumps(data[key], indent=2))
                    else:
                        details.append(f"## {key.capitalize()}")
                        details.append(str(data[key]))
                        
            # Look for assets
            if "assets" in data and isinstance(data["assets"], list):
                details.append("## Asset Analysis")
                for i, asset in enumerate(data["assets"]):
                    details.append(f"### {asset.get('symbol', f'Asset {i+1}')}")
                    for k, v in asset.items():
                        if k != "symbol":
                            details.append(f"- {k}: {v}")
        
        # If no structured data found, use content field or raw data
        if not details:
            if "content" in data:
                return data["content"]
            else:
                return json.dumps(data, indent=2)
                
        return "\n\n".join(details)
    
    def save_prompt(self, prompt: str, report_data: Dict) -> str:
        """
        Save generated prompt to file.
        
        Args:
            prompt: The generated prompt
            report_data: The report data used to generate the prompt
            
        Returns:
            Path to the saved prompt file
        """
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_type = report_data.get("report_type", "unknown")
        report_file = os.path.basename(report_data.get("report_path", "unknown"))
        base_name = os.path.splitext(report_file)[0]
        
        prompt_file = f"{timestamp}_{report_type}_{base_name}.txt"
        prompt_path = os.path.join(self.output_dir, prompt_file)
        
        # Save the prompt
        with open(prompt_path, 'w') as f:
            f.write(prompt)
            
        logger.info(f"Saved prompt to: {prompt_path}")
        return prompt_path
    
    def send_to_claude(self, prompt: str) -> Dict:
        """
        Send prompt to Claude API if API key is available.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Claude's response or error info
        """
        if not self.api_key:
            logger.warning("No API key provided. Cannot send to Claude.")
            return {"error": "No API key provided"}
            
        try:
            headers = {
                "x-api-key": self.api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.config.get("claude_model", "claude-3-5-sonnet-20240307"),
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 4000
            }
            
            response = requests.post(
                self.config.get("claude_api_endpoint", "https://api.anthropic.com/v1/messages"),
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("Successfully sent prompt to Claude and received response")
                return result
            else:
                logger.error(f"Error from Claude API: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}", "details": response.text}
                
        except Exception as e:
            logger.error(f"Exception when calling Claude API: {e}")
            return {"error": str(e)}
    
    def save_claude_response(self, response: Dict, prompt_path: str) -> Optional[str]:
        """
        Save Claude's response to file.
        
        Args:
            response: Claude's response
            prompt_path: Path to the prompt file
            
        Returns:
            Path to the saved response file or None if error
        """
        if "error" in response:
            logger.error(f"Cannot save Claude response due to error: {response['error']}")
            return None
            
        try:
            # Create response filename based on prompt filename
            prompt_file = os.path.basename(prompt_path)
            response_file = f"{os.path.splitext(prompt_file)[0]}_response.json"
            response_path = os.path.join(self.output_dir, response_file)
            
            # Save the response
            with open(response_path, 'w') as f:
                json.dump(response, f, indent=2)
                
            logger.info(f"Saved Claude response to: {response_path}")
            return response_path
        except Exception as e:
            logger.error(f"Error saving Claude response: {e}")
            return None
    
    def process_report(self, report_path: str) -> Dict:
        """
        Process a single report.
        
        Args:
            report_path: Path to the report file
            
        Returns:
            Processing results
        """
        results = {
            "report_path": report_path,
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Parse the report
            report_data = self.parse_report(report_path)
            results["report_type"] = report_data.get("report_type", "unknown")
            
            # Generate prompt
            prompt = self.generate_prompt(report_data)
            results["prompt_generated"] = True
            
            # Save prompt
            prompt_path = self.save_prompt(prompt, report_data)
            results["prompt_path"] = prompt_path
            
            # Send to Claude if API key is available
            if self.api_key:
                claude_response = self.send_to_claude(prompt)
                if "error" not in claude_response:
                    response_path = self.save_claude_response(claude_response, prompt_path)
                    results["claude_response_path"] = response_path
                    results["claude_response_received"] = True
                else:
                    results["claude_error"] = claude_response["error"]
                    results["claude_response_received"] = False
            
            # Mark as processed
            self.processed_reports.append(report_path)
            self._save_processed_reports()
            
            results["status"] = "completed"
            logger.info(f"Successfully processed report: {os.path.basename(report_path)}")
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            logger.error(f"Error processing report {report_path}: {e}")
            
        return results
    
    def run_once(self) -> List[Dict]:
        """
        Run the bridge once, processing any new reports.
        
        Returns:
            List of processing results
        """
        results = []
        
        # Scan for new reports
        new_reports = self.scan_for_new_reports()
        
        # Process each report
        for report_path in new_reports:
            result = self.process_report(report_path)
            results.append(result)
            
        return results
    
    def run_continuously(self, interval_seconds: Optional[int] = None):
        """
        Run the bridge continuously, checking for new reports at intervals.
        
        Args:
            interval_seconds: Seconds between checks. If None, use config value.
        """
        if interval_seconds is None:
            interval_seconds = self.config.get("scan_interval_seconds", 300)
            
        logger.info(f"Starting continuous operation with {interval_seconds}s interval")
        
        try:
            while True:
                results = self.run_once()
                if results:
                    logger.info(f"Processed {len(results)} reports")
                
                logger.info(f"Sleeping for {interval_seconds} seconds")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping.")
        except Exception as e:
            logger.error(f"Error in continuous operation: {e}")
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Bridge between EzioFinisher and Claude")
    parser.add_argument("--reports-dir", type=str, help="Directory containing EzioFinisher reports")
    parser.add_argument("--output-dir", type=str, help="Directory to save generated Claude prompts")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--api-key", type=str, help="Claude API key for direct communication")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, help="Seconds between checks in continuous mode")
    
    args = parser.parse_args()
    
    # Initialize bridge with provided args
    bridge_kwargs = {}
    if args.reports_dir:
        bridge_kwargs["reports_dir"] = args.reports_dir
    if args.output_dir:
        bridge_kwargs["output_dir"] = args.output_dir
    if args.config:
        bridge_kwargs["config_file"] = args.config
    if args.api_key:
        bridge_kwargs["api_key"] = args.api_key
        
    bridge = ClaudeSyncBridge(**bridge_kwargs)
    
    # Run bridge
    if args.continuous:
        bridge.run_continuously(args.interval)
    else:
        results = bridge.run_once()
        print(f"Processed {len(results)} reports")
        for result in results:
            status = result["status"]
            report = os.path.basename(result["report_path"])
            print(f"- {report}: {status}")


if __name__ == "__main__":
    main()