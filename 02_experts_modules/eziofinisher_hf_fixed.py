#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EzioFinisher - Secure Version for GitHub Codespaces
Autonomous Agent for Code Detection and Correction
Version: 2.0 - Secure & Flexible
"""

import os
import sys
import time
import json
import logging
import argparse
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Security: Use environment variables instead of hardcoded values
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "autogen_generated"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging with dynamic paths
LOG_FILE = LOG_DIR / "eziofinisher.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EzioFinisher")

class EzioFinisher:
    """Autonomous agent for code analysis and optimization."""
    
    def __init__(self, max_file_size=3000, file_limit=20, project_dir=None):
        """
        Initialize the EzioFinisher agent.
        
        Args:
            max_file_size: Maximum file size in characters
            file_limit: Limit of files to process
            project_dir: Project directory (defaults to current working directory)
        """
        self.max_file_size = max_file_size
        self.file_limit = file_limit
        
        # Flexible project directory
        self.project_dir = Path(project_dir) if project_dir else PROJECT_ROOT
        self.output_dir = OUTPUT_DIR
        
        # Security: Get token from environment
        self.hf_token = self._get_secure_token()
        
        # Initialize model with security checks
        self.pipe = self._initialize_model()
        
        # Cache of analyzed files
        self.analyzed_files = set()
        
        # Target file extensions
        self.target_extensions = {".py", ".js", ".html", ".css", ".md", ".txt", ".json", ".yaml", ".yml"}
        
        # Ignore directories
        self.ignore_dirs = {"venv", ".git", "__pycache__", "node_modules", "autogen_generated", "logs"}
        
        # Initialize improvement counter
        self.improvements_count = 0
        
        logger.info("EzioFinisher initialized successfully!")
        logger.info(f"Project directory: {self.project_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"File size limit: {self.max_file_size} characters")
        logger.info(f"File processing limit: {self.file_limit}")

    def _get_secure_token(self) -> str:
        """
        Securely get Hugging Face token from environment.
        
        Returns:
            str: HF token or empty string if not found
        """
        # Try multiple environment variable names
        token_vars = ['HUGGINGFACE_TOKEN', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN']
        
        for var in token_vars:
            token = os.getenv(var)
            if token:
                logger.info(f"Found HF token in environment variable: {var}")
                return token
        
        logger.warning("No Hugging Face token found in environment variables")
        logger.info("Model will work in offline mode or with public models only")
        return ""

    def _initialize_model(self) -> Optional[object]:
        """
        Initialize the local model with security checks.
        
        Returns:
            Pipeline object or None if initialization fails
        """
        try:
            from transformers import pipeline, AutoTokenizer
            
            logger.info("Initializing local model. This may take a few minutes...")
            
            # Use a smaller, more stable model
            model_name = "microsoft/CodeBERT-base"
            
            # Security: Only pass token if it exists
            token_kwargs = {"token": self.hf_token} if self.hf_token else {}
            
            # Initialize tokenizer first
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
                logger.info("Falling back to basic model without custom tokenizer")
                tokenizer = None
            
            # Configure pipeline with error handling
            pipeline_kwargs = {
                "task": "fill-mask",  # Changed to a more stable task
                "model": model_name,
                "tokenizer": tokenizer,
                **token_kwargs
            }
            
            pipe = pipeline(**pipeline_kwargs)
            logger.info("Local model initialized successfully")
            return pipe
            
        except ImportError:
            logger.warning("Transformers library not found. Install with 'pip install transformers'")
            logger.info("Continuing in static analysis mode")
            return None
        except Exception as e:
            logger.error(f"Error initializing local model: {e}")
            logger.info("Continuing in static analysis mode without model")
            return None

    def get_project_structure(self) -> Dict:
        """
        Map the project structure.
        
        Returns:
            Dict: Directory and file structure of the project.
        """
        structure = {"dirs": [], "files": []}
        
        try:
            for root, dirs, files in os.walk(self.project_dir):
                # Apply ignore directory filter
                dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
                
                relative_path = Path(root).relative_to(self.project_dir)
                relative_path_str = str(relative_path) if str(relative_path) != "." else ""
                
                # Add directory to structure
                if relative_path_str:
                    structure["dirs"].append(relative_path_str)
                
                # Add relevant files
                for file in files:
                    file_path = Path(file)
                    if file_path.suffix in self.target_extensions:
                        full_path = str(Path(relative_path_str) / file) if relative_path_str else file
                        structure["files"].append(full_path)
        
        except Exception as e:
            logger.error(f"Error scanning project structure: {e}")
            return {"dirs": [], "files": []}
        
        logger.info(f"Found {len(structure['files'])} files and {len(structure['dirs'])} directories")
        return structure

    def read_file(self, file_path: str) -> str:
        """
        Read file content safely.
        
        Args:
            file_path: File path relative to project directory
            
        Returns:
            str: File content or empty string on error
        """
        full_path = self.project_dir / file_path
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            try:
                with open(full_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                logger.warning(f"File {file_path} read with latin-1 encoding")
                return content
            except Exception as e:
                logger.error(f"Error reading file {file_path} with latin-1: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""

    def is_file_too_large(self, content: str) -> bool:
        """
        Check if file is too large for processing.
        
        Args:
            content: File content
            
        Returns:
            bool: True if file is too large
        """
        return len(content) > self.max_file_size

    def write_file(self, file_path: str, content: str) -> bool:
        """
        Write content to file in output directory.
        
        Args:
            file_path: Relative file path
            content: Content to write
            
        Returns:
            bool: True if file was written successfully
        """
        full_path = self.output_dir / file_path
        
        try:
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"File generated successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return False

    def analyze_code(self, file_content: str, file_path: str) -> Dict:
        """
        Analyze code using local model or static analysis.
        
        Args:
            file_content: File content
            file_path: File path for context
            
        Returns:
            Dict: Analysis result with problems and improvement suggestions
        """
        # Check file size
        if self.is_file_too_large(file_content):
            logger.warning(f"File {file_path} is too large ({len(file_content)} chars). "
                          f"Analyzing first {self.max_file_size} characters.")
            file_content = file_content[:self.max_file_size]
        
        # Use static analysis (more reliable for code analysis)
        return self.analyze_static(file_content, file_path)

    def analyze_static(self, file_content: str, file_path: str) -> Dict:
        """
        Perform static analysis without model.
        
        Args:
            file_content: File content
            file_path: File path
            
        Returns:
            Dict: Basic analysis
        """
        problems = []
        
        # Check for TODOs
        if "TODO" in file_content.upper():
            lines = file_content.split('\n')
            for i, line in enumerate(lines):
                if "TODO" in line.upper():
                    problems.append({
                        "type": "incomplete",
                        "line": str(i + 1),
                        "description": f"TODO found: {line.strip()}",
                        "severity": "medium"
                    })
        
        # Python-specific analysis
        if file_path.endswith('.py'):
            problems.extend(self._analyze_python_code(file_content))
        
        # JavaScript-specific analysis
        elif file_path.endswith('.js'):
            problems.extend(self._analyze_javascript_code(file_content))
        
        # General analysis for all files
        problems.extend(self._analyze_general_issues(file_content))
        
        return {
            "problems": problems,
            "corrected_code": file_content,
            "explanation": "Static analysis completed without AI model."
        }

    def _analyze_python_code(self, content: str) -> List[Dict]:
        """Analyze Python-specific issues."""
        problems = []
        lines = content.split('\n')
        
        # Check for empty functions
        empty_func_pattern = r'def\s+(\w+)[^:]*:\s*\n\s*(pass|return|#)'
        for match in re.finditer(empty_func_pattern, content):
            func_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            problems.append({
                "type": "incomplete",
                "line": str(line_num),
                "description": f"Empty or incomplete function: {func_name}",
                "severity": "high"
            })
        
        # Check for hardcoded credentials (security)
        security_patterns = [
            (r'token\s*=\s*["\'][^"\']*["\']', "Hardcoded token found"),
            (r'password\s*=\s*["\'][^"\']*["\']', "Hardcoded password found"),
            (r'api_key\s*=\s*["\'][^"\']*["\']', "Hardcoded API key found"),
            (r'secret\s*=\s*["\'][^"\']*["\']', "Hardcoded secret found")
        ]
        
        for pattern, message in security_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count('\n') + 1
                problems.append({
                    "type": "security",
                    "line": str(line_num),
                    "description": message,
                    "severity": "critical"
                })
        
        # Check for long functions
        func_starts = []
        current_func_start = None
        
        for i, line in enumerate(lines):
            if re.match(r'^\s*def\s+\w+\s*\(', line):
                if current_func_start is not None:
                    # Previous function ended
                    func_length = i - current_func_start
                    if func_length > 50:
                        func_name = re.search(r'def\s+(\w+)', lines[current_func_start]).group(1)
                        problems.append({
                            "type": "style",
                            "line": str(current_func_start + 1),
                            "description": f"Function too long ({func_length} lines): {func_name}",
                            "severity": "medium"
                        })
                current_func_start = i
        
        return problems

    def _analyze_javascript_code(self, content: str) -> List[Dict]:
        """Analyze JavaScript-specific issues."""
        problems = []
        
        # Check for console.log (should be removed in production)
        for match in re.finditer(r'console\.log\s*\(', content):
            line_num = content[:match.start()].count('\n') + 1
            problems.append({
                "type": "style",
                "line": str(line_num),
                "description": "console.log found - consider removing for production",
                "severity": "low"
            })
        
        # Check for var usage (prefer let/const)
        for match in re.finditer(r'\bvar\s+\w+', content):
            line_num = content[:match.start()].count('\n') + 1
            problems.append({
                "type": "style",
                "line": str(line_num),
                "description": "Use 'let' or 'const' instead of 'var'",
                "severity": "medium"
            })
        
        return problems

    def _analyze_general_issues(self, content: str) -> List[Dict]:
        """Analyze general issues in any file type."""
        problems = []
        
        # Check for very long lines
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 120:
                problems.append({
                    "type": "style",
                    "line": str(i + 1),
                    "description": f"Line too long ({len(line)} characters)",
                    "severity": "low"
                })
        
        # Check for trailing whitespace
        for i, line in enumerate(lines):
            if line.rstrip() != line:
                problems.append({
                    "type": "style",
                    "line": str(i + 1),
                    "description": "Trailing whitespace found",
                    "severity": "low"
                })
        
        return problems

    def analyze_and_fix_file(self, file_path: str) -> bool:
        """
        Analyze and fix a specific file.
        
        Args:
            file_path: Relative file path
            
        Returns:
            bool: True if file was analyzed successfully
        """
        logger.info(f"Analyzing file: {file_path}")
        
        # Read file content
        file_content = self.read_file(file_path)
        if not file_content:
            return False
        
        # Add to analyzed files set
        self.analyzed_files.add(file_path)
        
        # Execute analysis
        analysis = self.analyze_code(file_content, file_path)
        
        # Check if problems were identified
        if not analysis["problems"]:
            logger.info(f"No problems identified in {file_path}")
            return True
        
        # Log found problems
        for problem in analysis["problems"]:
            logger.info(f"Problem in {file_path}, line {problem.get('line', 'N/A')}: "
                       f"{problem.get('description', 'No description')} "
                       f"(Severity: {problem.get('severity', 'N/A')})")
        
        # Save original file and report
        output_path = Path("problems_found") / file_path
        self.write_file(str(output_path), file_content)
        self.improvements_count += 1
        
        # Generate report file
        report_path = Path("reports") / f"{Path(file_path).name}.txt"
        report_content = f"Problem Report for {file_path}\n"
        report_content += "=" * 80 + "\n\n"
        
        for problem in analysis["problems"]:
            report_content += f"- TYPE: {problem.get('type', 'unknown')}\n"
            report_content += f"  LINE: {problem.get('line', 'N/A')}\n"
            report_content += f"  SEVERITY: {problem.get('severity', 'medium')}\n"
            report_content += f"  DESCRIPTION: {problem.get('description', 'No description')}\n\n"
        
        self.write_file(str(report_path), report_content)
        
        return True

    def create_report_index(self):
        """Create an index file for generated reports."""
        try:
            report_dir = self.output_dir / "reports"
            if not report_dir.exists():
                return
            
            reports = [f for f in report_dir.iterdir() if f.suffix == '.txt']
            if not reports:
                return
            
            index_content = "# Problem Reports Index\n\n"
            
            for report in reports:
                file_name = report.stem
                index_content += f"- [{file_name}](reports/{report.name})\n"
            
            self.write_file("report_index.md", index_content)
            logger.info("Report index created successfully.")
            
        except Exception as e:
            logger.error(f"Error creating report index: {e}")

    def create_missing_files(self) -> int:
        """
        Create basic files that seem to be missing.
        
        Returns:
            int: Number of files created
        """
        essential_files = [
            {
                "filename": "README.md",
                "exists": (self.project_dir / "README.md").exists(),
                "content": """# EZIO Financial AI Trading System

Advanced multi-agent platform for financial analysis and trading.

## Features

- Automatic problem detection
- Code correction
- Script generation
- GitHub Codespaces support

## Quick Start

```bash
python start_ezio_codespaces.py
```

## Environment Variables

- `HUGGINGFACE_TOKEN`: Optional HF token for model access
- `OPENAI_API_KEY`: Optional OpenAI API key

## License

MIT License
"""
            },
            {
                "filename": "requirements.txt",
                "exists": (self.project_dir / "requirements.txt").exists(),
                "content": """# Core dependencies
transformers>=4.30.0
torch>=2.0.0
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.24.0
requests>=2.28.0

# Optional AI/ML dependencies
autogen>=0.1.0
langchain>=0.1.0
openai>=1.0.0

# Development dependencies
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
"""
            }
        ]
        
        count = 0
        for file_info in essential_files:
            if not file_info["exists"]:
                logger.info(f"Creating missing essential file: {file_info['filename']}")
                
                if self.write_file(file_info["filename"], file_info["content"]):
                    count += 1
                    self.improvements_count += 1
        
        return count

    def run(self, max_iterations: int = 1) -> Dict:
        """
        Execute the analysis and correction process.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Dict: Summary of results
        """
        start_time = time.time()
        logger.info(f"Starting EzioFinisher with {max_iterations} max iterations")
        
        results = {
            "files_analyzed": 0,
            "problems_corrected": 0,
            "files_created": 0,
            "errors": 0
        }
        
        # Create output directories
        (self.output_dir / "problems_found").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        for iteration in range(max_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
            
            # Get project structure
            structure = self.get_project_structure()
            
            # Limit number of files
            limited_files = structure["files"][:self.file_limit]
            logger.info(f"Limiting analysis to {len(limited_files)} of {len(structure['files'])} files")
            
            # Analyze and fix existing files
            for file_path in limited_files:
                if file_path in self.analyzed_files:
                    continue
                
                success = self.analyze_and_fix_file(file_path)
                results["files_analyzed"] += 1
                if not success:
                    results["errors"] += 1
            
            # Create missing basic files
            created_count = self.create_missing_files()
            results["files_created"] += created_count
            
            # Update corrected problems count
            results["problems_corrected"] = self.improvements_count
            
            logger.info(f"Iteration {iteration + 1} completed. Partial summary: {results}")
        
        # Create report index
        self.create_report_index()
        
        elapsed_time = time.time() - start_time
        logger.info(f"EzioFinisher completed in {elapsed_time:.2f} seconds")
        logger.info(f"Final summary: {results}")
        
        return results

def main():
    """Main function to execute EzioFinisher as a script."""
    parser = argparse.ArgumentParser(
        description="EzioFinisher - Autonomous agent for code correction (Secure Version)"
    )
    parser.add_argument("--max-iterations", type=int, default=1, 
                       help="Maximum number of iterations")
    parser.add_argument("--max-file-size", type=int, default=3000, 
                       help="Maximum file size in characters")
    parser.add_argument("--file-limit", type=int, default=20, 
                       help="Limit of files to process")
    parser.add_argument("--project-dir", type=str, 
                       help="Project directory path")
    parser.add_argument("--static-only", action="store_true", 
                       help="Run only static analysis (no AI model)")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run EzioFinisher
        ezio = EzioFinisher(
            max_file_size=args.max_file_size,
            file_limit=args.file_limit,
            project_dir=args.project_dir
        )
        
        # Run analysis
        results = ezio.run(max_iterations=args.max_iterations)
        
        # Print final summary
        print("\n" + "=" * 50)
        print("EZIO FINISHER SUMMARY")
        print("=" * 50)
        print(f"Files analyzed: {results['files_analyzed']}")
        print(f"Problems found: {results['problems_corrected']}")
        print(f"Files created: {results['files_created']}")
        print(f"Errors: {results['errors']}")
        print("=" * 50)
        print(f"Complete log available at: {LOG_FILE}")
        print(f"Generated files saved in: {OUTPUT_DIR}")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        print(f"Error executing EzioFinisher: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())