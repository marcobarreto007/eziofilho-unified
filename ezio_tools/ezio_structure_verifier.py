#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EzioStructureVerifier - Project Structure Analysis Tool

This script analyzes the directory and file structure of the EzioFilhoUnified project
and generates a report on what's present, what's missing, and suggested actions.
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import logging
import json
import platform
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EzioStructureVerifier")

@dataclass
class StructureComponent:
    """Dataclass representing a component in the project structure."""
    name: str
    type: str  # "directory" or "file"
    required: bool
    description: str
    parent: Optional[str] = None
    children: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class VerificationResult:
    """Dataclass representing the verification result for a component."""
    component: StructureComponent
    exists: bool
    is_complete: bool
    missing_children: List[str] = None
    
    def __post_init__(self):
        if self.missing_children is None:
            self.missing_children = []

class EzioStructureVerifier:
    """
    Analyzes the directory and file structure of the EzioFilhoUnified project.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the verifier with the project base path.
        
        Args:
            base_path: The base path of the EzioFilhoUnified project.
        """
        self.base_path = base_path or self._detect_base_path()
        if not self.base_path:
            logger.error("Could not detect the project base path. Please provide it explicitly.")
            sys.exit(1)
            
        self.expected_structure = self._define_expected_structure()
        self.verification_results = []
        self.completeness_score = 0.0
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, str]:
        """
        Get system information.
        
        Returns:
            A dictionary with system information.
        """
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "username": os.getenv("USERNAME") or os.getenv("USER") or "unknown",
            "hostname": platform.node()
        }
        
    def _detect_base_path(self) -> str:
        """
        Attempt to detect the project base path.
        
        Returns:
            The detected base path, or None if not found.
        """
        # Try different common parent directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            current_dir,
            os.path.dirname(current_dir),
            os.path.join(os.path.dirname(current_dir), "EzioFilhoUnified"),
            "C:\\Users\\anapa\\SuperIA\\EzioFilhoUnified"  # From the example command
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and self._looks_like_project_root(path):
                return path
                
        return None
        
    def _looks_like_project_root(self, path: str) -> bool:
        """
        Check if the given path looks like the project root directory.
        
        Args:
            path: The path to check.
            
        Returns:
            True if the path looks like the project root, False otherwise.
        """
        # Check for some common directories/files that would exist in the project root
        indicators = ['ezio_experts', 'core', 'models', 'experts', 'data', 'tests']
        found = 0
        for indicator in indicators:
            if os.path.exists(os.path.join(path, indicator)):
                found += 1
        # If at least 2 indicators are found, consider it a project root
        return found >= 2
        
    def _define_expected_structure(self) -> Dict[str, StructureComponent]:
        """
        Define the expected project structure.
        
        Returns:
            A dictionary mapping component names to StructureComponent objects.
        """
        components = {}
        
        # Top-level directories
        components["root"] = StructureComponent(
            name="root",
            type="directory",
            required=True,
            description="Project root directory",
            children=["core", "models", "experts", "ezio_experts", "data", "tests", 
                      "ezio_tools", "autogen_generated", "gui", "venv_ezio"]
        )
        
        # Core components
        components["core"] = StructureComponent(
            name="core",
            type="directory",
            required=True,
            description="Core system components",
            parent="root",
            children=["quantum_moe_core", "model_router", "rlhf_trainer", "langchain_integration", "langgraph_flows"]
        )
        
        components["quantum_moe_core"] = StructureComponent(
            name="quantum_moe_core",
            type="directory",
            required=True,
            description="Quantum MoE Core implementation",
            parent="core",
            children=["quantum_moe_core.py", "moe_dispatcher.py"]
        )
        
        components["quantum_moe_core.py"] = StructureComponent(
            name="quantum_moe_core.py",
            type="file",
            required=True,
            description="Quantum MoE Core implementation file",
            parent="quantum_moe_core"
        )
        
        components["moe_dispatcher.py"] = StructureComponent(
            name="moe_dispatcher.py",
            type="file",
            required=True,
            description="MoE dispatcher implementation file",
            parent="quantum_moe_core"
        )
        
        components["model_router"] = StructureComponent(
            name="model_router",
            type="directory",
            required=True,
            description="Model routing system",
            parent="core",
            children=["model_router.py", "router_config.py"]
        )
        
        components["model_router.py"] = StructureComponent(
            name="model_router.py",
            type="file",
            required=True,
            description="Model router implementation file",
            parent="model_router"
        )
        
        components["router_config.py"] = StructureComponent(
            name="router_config.py",
            type="file",
            required=True,
            description="Router configuration file",
            parent="model_router"
        )
        
        components["rlhf_trainer"] = StructureComponent(
            name="rlhf_trainer",
            type="directory",
            required=True,
            description="RLHF training system",
            parent="core",
            children=["rlhf_trainer.py", "reward_model.py"]
        )
        
        components["rlhf_trainer.py"] = StructureComponent(
            name="rlhf_trainer.py",
            type="file",
            required=True,
            description="RLHF trainer implementation file",
            parent="rlhf_trainer"
        )
        
        components["reward_model.py"] = StructureComponent(
            name="reward_model.py",
            type="file",
            required=True,
            description="Reward model implementation file",
            parent="rlhf_trainer"
        )
        
        # Models
        components["models"] = StructureComponent(
            name="models",
            type="directory",
            required=True,
            description="Local models storage",
            parent="root",
            children=["phi-2", "phi-3"]
        )
        
        components["phi-2"] = StructureComponent(
            name="phi-2",
            type="directory",
            required=False,
            description="Phi-2 model directory",
            parent="models"
        )
        
        components["phi-3"] = StructureComponent(
            name="phi-3",
            type="directory",
            required=False,
            description="Phi-3 model directory",
            parent="models"
        )
        
        # Experts
        components["experts"] = StructureComponent(
            name="experts",
            type="directory",
            required=True,
            description="System experts directory",
            parent="root",
            children=["base_expert.py"]
        )
        
        components["base_expert.py"] = StructureComponent(
            name="base_expert.py",
            type="file",
            required=True,
            description="Base expert implementation file",
            parent="experts"
        )
        
        components["ezio_experts"] = StructureComponent(
            name="ezio_experts",
            type="directory",
            required=True,
            description="Custom experts directory",
            parent="root",
            children=["fallback_data_expert", "cache_expert", "sentiment_expert", "news_oracle"]
        )
        
        components["fallback_data_expert"] = StructureComponent(
            name="fallback_data_expert",
            type="directory",
            required=True,
            description="Fallback data expert",
            parent="ezio_experts",
            children=["fallback_data_expert.py"]
        )
        
        components["fallback_data_expert.py"] = StructureComponent(
            name="fallback_data_expert.py",
            type="file",
            required=True,
            description="Fallback data expert implementation file",
            parent="fallback_data_expert"
        )
        
        components["cache_expert"] = StructureComponent(
            name="cache_expert",
            type="directory",
            required=True,
            description="Cache expert",
            parent="ezio_experts",
            children=["cache_expert.py"]
        )
        
        components["cache_expert.py"] = StructureComponent(
            name="cache_expert.py",
            type="file",
            required=True,
            description="Cache expert implementation file",
            parent="cache_expert"
        )
        
        components["sentiment_expert"] = StructureComponent(
            name="sentiment_expert",
            type="directory",
            required=True,
            description="Sentiment analysis expert",
            parent="ezio_experts",
            children=["sentiment_expert.py"]
        )
        
        components["sentiment_expert.py"] = StructureComponent(
            name="sentiment_expert.py",
            type="file",
            required=True,
            description="Sentiment expert implementation file",
            parent="sentiment_expert"
        )
        
        components["news_oracle"] = StructureComponent(
            name="news_oracle",
            type="directory",
            required=True,
            description="News oracle expert",
            parent="ezio_experts",
            children=["news_oracle.py"]
        )
        
        components["news_oracle.py"] = StructureComponent(
            name="news_oracle.py",
            type="file",
            required=True,
            description="News oracle implementation file",
            parent="news_oracle"
        )
        
        # Data
        components["data"] = StructureComponent(
            name="data",
            type="directory",
            required=True,
            description="Data storage directory",
            parent="root",
            children=["cache", "fallback_data"]
        )
        
        components["cache"] = StructureComponent(
            name="cache",
            type="directory",
            required=True,
            description="Cache data directory",
            parent="data"
        )
        
        components["fallback_data"] = StructureComponent(
            name="fallback_data",
            type="directory",
            required=True,
            description="Fallback data directory",
            parent="data"
        )
        
        # Tests
        components["tests"] = StructureComponent(
            name="tests",
            type="directory",
            required=True,
            description="Test directory",
            parent="root",
            children=["test_experts", "test_core", "test_gui"]
        )
        
        components["test_experts"] = StructureComponent(
            name="test_experts",
            type="directory",
            required=True,
            description="Expert tests directory",
            parent="tests"
        )
        
        components["test_core"] = StructureComponent(
            name="test_core",
            type="directory",
            required=True,
            description="Core tests directory",
            parent="tests"
        )
        
        components["test_gui"] = StructureComponent(
            name="test_gui",
            type="directory",
            required=True,
            description="GUI tests directory",
            parent="tests"
        )
        
        # Tools
        components["ezio_tools"] = StructureComponent(
            name="ezio_tools",
            type="directory",
            required=True,
            description="Project tools",
            parent="root",
            children=["ezio_finisher.py", "ezio_structure_verifier.py"]
        )
        
        components["ezio_finisher.py"] = StructureComponent(
            name="ezio_finisher.py",
            type="file",
            required=True,
            description="Code audit tool",
            parent="ezio_tools"
        )
        
        components["ezio_structure_verifier.py"] = StructureComponent(
            name="ezio_structure_verifier.py",
            type="file",
            required=True,
            description="Structure verification tool (this script)",
            parent="ezio_tools"
        )
        
        # AutoGen
        components["autogen_generated"] = StructureComponent(
            name="autogen_generated",
            type="directory",
            required=True,
            description="AutoGen generated code",
            parent="root",
            children=["autogen_config.json"]
        )
        
        components["autogen_config.json"] = StructureComponent(
            name="autogen_config.json",
            type="file",
            required=True,
            description="AutoGen configuration file",
            parent="autogen_generated"
        )
        
        # GUI
        components["gui"] = StructureComponent(
            name="gui",
            type="directory",
            required=True,
            description="GUI components",
            parent="root",
            children=["magnetic_ui", "localgui_server"]
        )
        
        components["magnetic_ui"] = StructureComponent(
            name="magnetic_ui",
            type="directory",
            required=True,
            description="Magnetic UI framework",
            parent="gui",
            children=["magnetic_ui.py"]
        )
        
        components["magnetic_ui.py"] = StructureComponent(
            name="magnetic_ui.py",
            type="file",
            required=True,
            description="Magnetic UI implementation file",
            parent="magnetic_ui"
        )
        
        components["localgui_server"] = StructureComponent(
            name="localgui_server",
            type="directory",
            required=True,
            description="Local GUI server",
            parent="gui",
            children=["server.py"]
        )
        
        components["server.py"] = StructureComponent(
            name="server.py",
            type="file",
            required=True,
            description="Local GUI server implementation file",
            parent="localgui_server"
        )
        
        # Virtual environment
        components["venv_ezio"] = StructureComponent(
            name="venv_ezio",
            type="directory",
            required=True,
            description="Python virtual environment",
            parent="root"
        )
        
        # LangChain/LangGraph integration
        components["langchain_integration"] = StructureComponent(
            name="langchain_integration",
            type="directory",
            required=True,
            description="LangChain integration components",
            parent="core",
            children=["langchain_executor.py"]
        )
        
        components["langchain_executor.py"] = StructureComponent(
            name="langchain_executor.py",
            type="file",
            required=True,
            description="LangChain executor implementation file",
            parent="langchain_integration"
        )
        
        components["langgraph_flows"] = StructureComponent(
            name="langgraph_flows",
            type="directory",
            required=True,
            description="LangGraph flow definitions",
            parent="core",
            children=["graph_definitions.py"]
        )
        
        components["graph_definitions.py"] = StructureComponent(
            name="graph_definitions.py",
            type="file",
            required=True,
            description="LangGraph flow definitions file",
            parent="langgraph_flows"
        )
        
        return components
    
    def verify_structure(self) -> None:
        """
        Verify the project structure against the expected structure.
        """
        logger.info(f"Verifying project structure in: {self.base_path}")
        
        # Walk through expected structure and check existence
        total_components = 0
        existing_components = 0
        complete_components = 0
        
        for name, component in self.expected_structure.items():
            total_components += 1
            
            if name == "root":
                path = self.base_path
            else:
                parent_path = self.base_path
                if component.parent and component.parent != "root":
                    parent_component = self.expected_structure[component.parent]
                    if parent_component.parent and parent_component.parent != "root":
                        # Handle nested components
                        parent_parent = self.expected_structure[parent_component.parent]
                        parent_path = os.path.join(self.base_path, parent_component.parent, component.parent)
                    else:
                        parent_path = os.path.join(self.base_path, component.parent)
                
                path = os.path.join(parent_path, name)
            
            exists = os.path.exists(path)
            if exists:
                existing_components += 1
            
            # Check if it's complete (all required children exist)
            is_complete = exists
            missing_children = []
            
            if exists and component.children:
                for child in component.children:
                    child_component = self.expected_structure.get(child)
                    if child_component and child_component.required:
                        child_path = os.path.join(path, child)
                        if not os.path.exists(child_path):
                            is_complete = False
                            missing_children.append(child)
            
            if is_complete:
                complete_components += 1
                
            self.verification_results.append(VerificationResult(
                component=component,
                exists=exists,
                is_complete=is_complete,
                missing_children=missing_children
            ))
        
        # Calculate completeness score
        if total_components > 0:
            self.completeness_score = (complete_components / total_components) * 100
        
        logger.info(f"Verification complete. Completeness score: {self.completeness_score:.2f}%")
    
    def generate_action_items(self) -> List[str]:
        """
        Generate action items based on verification results.
        
        Returns:
            A list of action items.
        """
        action_items = []
        
        # Add missing components
        for result in self.verification_results:
            component = result.component
            
            if not result.exists and component.required:
                if component.type == "directory":
                    action_items.append(f"Create directory: {component.name} - {component.description}")
                else:
                    action_items.append(f"Create file: {component.name} - {component.description}")
            
            # Add missing children
            if result.exists and not result.is_complete:
                for child in result.missing_children:
                    child_component = self.expected_structure.get(child)
                    if child_component:
                        if child_component.type == "directory":
                            action_items.append(f"Create directory: {child} in {component.name} - {child_component.description}")
                        else:
                            action_items.append(f"Create file: {child} in {component.name} - {child_component.description}")
        
        # Check for critical functionality
        critical_systems = {
            "AutoGen integration": ["autogen_generated", "autogen_config.json"],
            "GUI interface": ["gui", "magnetic_ui", "localgui_server"],
            "Required experts": ["fallback_data_expert", "cache_expert", "sentiment_expert", "news_oracle"],
            "LangGraph/LangChain": ["langchain_integration", "langgraph_flows"],
            "Model Router": ["model_router"],
            "RLHF Trainer": ["rlhf_trainer"],
            "News Oracle": ["news_oracle"],
            "Fallback Data": ["fallback_data"],
            "Cache": ["cache"],
            "Tests for experts": ["test_experts"]
        }
        
        for system_name, component_names in critical_systems.items():
            missing_components = []
            for component_name in component_names:
                component_exists = False
                for result in self.verification_results:
                    if result.component.name == component_name and result.exists:
                        component_exists = True
                        break
                
                if not component_exists:
                    missing_components.append(component_name)
            
            if missing_components:
                action_items.append(f"Implement {system_name}: missing {', '.join(missing_components)}")
        
        return action_items
    
    def generate_markdown_report(self) -> str:
        """
        Generate a Markdown report of the verification results.
        
        Returns:
            A string containing the Markdown report.
        """
        report = []
        report.append("# EzioFilhoUnified Structure Verification Report")
        report.append(f"\n**Generated:** {self.timestamp}")
        report.append(f"**User:** {self.system_info['username']}")
        report.append(f"**System:** {self.system_info['platform']}")
        report.append(f"**Python Version:** {self.system_info['python_version']}")
        report.append(f"**Project Path:** {self.base_path}")
        report.append(f"\n## Overall Completeness: {self.completeness_score:.2f}%")
        
        # Present Components
        report.append("\n## ✅ Present and Correct Components")
        present_count = 0
        for result in self.verification_results:
            if result.exists and result.is_complete:
                present_count += 1
                report.append(f"- **{result.component.name}**: {result.component.description}")
        
        if present_count == 0:
            report.append("- *No components are completely present.*")
            
        # Incomplete Components
        report.append("\n## ⚠️ Incomplete Components")
        incomplete_count = 0
        for result in self.verification_results:
            if result.exists and not result.is_complete:
                incomplete_count += 1
                report.append(f"- **{result.component.name}**: {result.component.description}")
                report.append(f"  - Missing children: {', '.join(result.missing_children)}")
        
        if incomplete_count == 0:
            report.append("- *No incomplete components.*")
            
        # Missing Components
        report.append("\n## ❌ Missing Components")
        missing_count = 0
        for result in self.verification_results:
            if not result.exists and result.component.required:
                missing_count += 1
                report.append(f"- **{result.component.name}**: {result.component.description}")
        
        if missing_count == 0:
            report.append("- *No required components are missing.*")
            
        # Action Items
        report.append("\n## 📋 Suggested Action Items")
        action_items = self.generate_action_items()
        if action_items:
            for i, item in enumerate(action_items, 1):
                report.append(f"{i}. {item}")
        else:
            report.append("- *No action items.*")
            
        # Critical Functionality Check
        report.append("\n## 🔍 Critical Functionality Assessment")
        
        critical_systems = {
            "AutoGen Integration": ["autogen_generated", "autogen_config.json"],
            "GUI Interface": ["gui", "magnetic_ui", "localgui_server"],
            "Required Experts": ["fallback_data_expert", "cache_expert", "sentiment_expert", "news_oracle"],
            "LangGraph/LangChain": ["langchain_integration", "langgraph_flows"],
            "Model Router": ["model_router"],
            "RLHF Trainer": ["rlhf_trainer"],
            "Fallback Data & Cache": ["fallback_data", "cache"],
            "Tests": ["test_experts", "test_core", "test_gui"]
        }
        
        for system_name, component_names in critical_systems.items():
            total = len(component_names)
            existing = 0
            
            for component_name in component_names:
                for result in self.verification_results:
                    if result.component.name == component_name and result.exists:
                        existing += 1
                        break
            
            percentage = (existing / total) * 100
            status = "✅" if percentage == 100 else "⚠️" if percentage > 0 else "❌"
            report.append(f"- {status} **{system_name}**: {existing}/{total} components ({percentage:.0f}%)")
        
        return "\n".join(report)
    
    def generate_text_report(self) -> str:
        """
        Generate a plain text report of the verification results.
        
        Returns:
            A string containing the plain text report.
        """
        # Convert markdown to plain text by removing # and * characters
        markdown = self.generate_markdown_report()
        text = markdown.replace("# ", "").replace("## ", "").replace("**", "").replace("*", "")
        return text
    
    def generate_html_report(self) -> str:
        """
        Generate an HTML report of the verification results.
        
        Returns:
            A string containing the HTML report.
        """
        markdown = self.generate_markdown_report()
        
        # Simple markdown to HTML conversion
        html = markdown.replace("# ", "<h1>").replace("\n## ", "</p><h2>")
        html = html.replace("**", "<strong>").replace("*", "<em>")
        html = html.replace("\n- ", "</p><p>• ").replace("\n", "<br>")
        html = html.replace("<h1>", "</p><h1>").replace("<h2>", "</h2><h2>")
        html = "<html><head><title>EzioFilhoUnified Structure Verification Report</title></head><body>" + html + "</p></body></html>"
        
        # Clean up
        html = html.replace("<p></p>", "").replace("<br></p>", "</p>")
        html = html.replace("</h1><br>", "</h1>").replace("</h2><br>", "</h2>")
        html = html.replace("<p><br>", "<p>")
        
        return html
    
    def save_reports(self, output_dir: str = None) -> Dict[str, str]:
        """
        Save reports to files.
        
        Args:
            output_dir: The directory to save the reports to. If None, use the project root.
            
        Returns:
            A dictionary mapping report format to file path.
        """
        if output_dir is None:
            output_dir = os.path.join(self.base_path, "ezio_reports")
            
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        # Markdown report
        md_path = os.path.join(output_dir, f"structure_report_{timestamp}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report())
        report_files["markdown"] = md_path
        
        # Text report
        txt_path = os.path.join(output_dir, f"structure_report_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.generate_text_report())
        report_files["text"] = txt_path
        
        # HTML report
        html_path = os.path.join(output_dir, f"structure_report_{timestamp}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self.generate_html_report())
        report_files["html"] = html_path
        
        logger.info(f"Reports saved to {output_dir}")
        return report_files

    def print_summary(self) -> None:
        """
        Print a summary of the verification results to the console.
        """
        print("\n" + "="*80)
        print(f"EzioFilhoUnified Structure Verification Summary")
        print("="*80)
        
        print(f"\nOverall Completeness: {self.completeness_score:.2f}%")
        
        # Count components by state
        present_complete = 0
        present_incomplete = 0
        missing_required = 0
        
        for result in self.verification_results:
            if result.exists and result.is_complete:
                present_complete += 1
            elif result.exists and not result.is_complete:
                present_incomplete += 1
            elif not result.exists and result.component.required:
                missing_required += 1
        
        print(f"\nComponents Status:")
        print(f"✅ Present and complete: {present_complete}")
        print(f"⚠️ Present but incomplete: {present_incomplete}")
        print(f"❌ Missing and required: {missing_required}")
        
        # Print a few action items as preview
        action_items = self.generate_action_items()
        if action_items:
            print(f"\nTop {min(5, len(action_items))} Action Items (of {len(action_items)} total):")
            for i, item in enumerate(action_items[:5], 1):
                print(f"{i}. {item}")
            
            if len(action_items) > 5:
                print(f"... and {len(action_items) - 5} more items.")
        
        # Print report locations
        print("\nDetailed reports have been generated.")
        print("Run with --save-reports to save detailed reports to files.")
        print("="*80 + "\n")

def main():
    """
    Main function to run the EzioStructureVerifier.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify the structure of the EzioFilhoUnified project.")
    parser.add_argument("--path", type=str, help="Path to the project root.")
    parser.add_argument("--save-reports", action="store_true", help="Save reports to files.")
    parser.add_argument("--output-dir", type=str, help="Directory to save reports to.")
    args = parser.parse_args()
    
    verifier = EzioStructureVerifier(base_path=args.path)
    verifier.verify_structure()
    verifier.print_summary()
    
    if args.save_reports:
        report_files = verifier.save_reports(output_dir=args.output_dir)
        print(f"\nReports saved:")
        for format_name, file_path in report_files.items():
            print(f"- {format_name.capitalize()}: {file_path}")
    else:
        # Always print the markdown report to console
        print("\n" + verifier.generate_markdown_report())

if __name__ == "__main__":
    main()