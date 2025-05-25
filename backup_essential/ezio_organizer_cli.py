#!/usr/bin/env python3
"""
EZIO PROJECT ORGANIZER CLI - COPILOT AUTOMATION
==================================================
Generates automated commands for project cleanup and organization.
Execute this script and copy/paste the generated commands to copilot.
"""

import os
import json
from datetime import datetime

class EzioOrganizerCLI:
    def __init__(self, base_path="C:\\Users\\anapa\\eziofilho-unified"):
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.commands = []
        self.report = {
            "timestamp": self.timestamp,
            "actions_performed": [],
            "files_moved": 0,
            "files_deleted": 0,
            "structure_created": []
        }
    
    def add_command(self, cmd, description=""):
        """Add command to execution list"""
        self.commands.append({
            "command": cmd,
            "description": description,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        if description:
            self.report["actions_performed"].append(description)
    
    def generate_new_structure(self):
        """Generate commands to create organized folder structure"""
        print("🏗️ GENERATING NEW STRUCTURE COMMANDS...")
        
        new_dirs = [
            # Main organization
            "01_core_system",
            "02_experts_modules", 
            "03_models_storage",
            "04_configuration",
            "05_testing_validation",
            "06_tools_utilities",
            "07_documentation",
            "08_examples_demos",
            "09_data_cache",
            "10_deployment",
            
            # Subdirectories
            "01_core_system\\main_runners",
            "01_core_system\\orchestrators", 
            "01_core_system\\base_classes",
            "02_experts_modules\\financial_experts",
            "02_experts_modules\\ai_integrations",
            "02_experts_modules\\specialized_tools",
            "03_models_storage\\local_models",
            "03_models_storage\\model_configs",
            "04_configuration\\project_configs",
            "04_configuration\\requirements_files",
            "05_testing_validation\\unit_tests",
            "05_testing_validation\\integration_tests",
            "05_testing_validation\\test_results",
            "06_tools_utilities\\gpu_tools",
            "06_tools_utilities\\monitoring",
            "07_documentation\\reports",
            "07_documentation\\guides",
            "08_examples_demos\\autogen_examples",
            "08_examples_demos\\working_demos",
            "09_data_cache\\shared_cache",
            "09_data_cache\\fallback_data",
            "10_deployment\\batch_files",
            "10_deployment\\production_ready"
        ]
        
        for dir_path in new_dirs:
            full_path = f"{self.base_path}\\{dir_path}"
            self.add_command(f'mkdir "{full_path}"', f"Create directory: {dir_path}")
            self.report["structure_created"].append(dir_path)
    
    def generate_file_organization_commands(self):
        """Generate commands to move files to appropriate locations"""
        print("🗂️ GENERATING FILE ORGANIZATION COMMANDS...")
        
        # Main runners and orchestrators
        main_files = [
            "main.py", "main_fixed.py", "main_autogen.py", "run_system.py", 
            "run_unified.py", "run_local.py", "run_with_auto_discovery.py"
        ]
        for file in main_files:
            self.add_command(
                f'move "{self.base_path}\\{file}" "{self.base_path}\\01_core_system\\main_runners\\"',
                f"Move main runner: {file}"
            )
        
        # Orchestrators
        orchestrators = [
            "ezio_orchestrator.py", "unified_orchestrator.py"
        ]
        for file in orchestrators:
            self.add_command(
                f'move "{self.base_path}\\{file}" "{self.base_path}\\01_core_system\\orchestrators\\"',
                f"Move orchestrator: {file}"
            )
        
        # Expert systems
        expert_files = [
            "expert_fingpt.py", "eziofinisher_*.py", "advanced_model_manager.py",
            "direct_chat.py", "simple_chat.py"
        ]
        for pattern in expert_files:
            if "*" in pattern:
                # For wildcards, generate a for loop command
                self.add_command(
                    f'for %f in ("{self.base_path}\\{pattern}") do move "%f" "{self.base_path}\\02_experts_modules\\financial_experts\\"',
                    f"Move expert files: {pattern}"
                )
            else:
                self.add_command(
                    f'move "{self.base_path}\\{pattern}" "{self.base_path}\\02_experts_modules\\financial_experts\\"',
                    f"Move expert file: {pattern}"
                )
        
        # Configuration files
        config_files = [
            "config_*.py", "config_*.json", "*.json", "configure_*.py", "criar_config.py"
        ]
        for pattern in config_files:
            if "*" in pattern:
                self.add_command(
                    f'for %f in ("{self.base_path}\\{pattern}") do move "%f" "{self.base_path}\\04_configuration\\project_configs\\"',
                    f"Move config files: {pattern}"
                )
        
        # Requirements files
        req_files = [
            "requirements*.txt", "requirements*.txt"
        ]
        for pattern in req_files:
            self.add_command(
                f'for %f in ("{self.base_path}\\{pattern}") do move "%f" "{self.base_path}\\04_configuration\\requirements_files\\"',
                f"Move requirements: {pattern}"
            )
        
        # Test files
        test_files = [
            "test_*.py", "run_*test*.py", "*_test.py"
        ]
        for pattern in test_files:
            self.add_command(
                f'for %f in ("{self.base_path}\\{pattern}") do move "%f" "{self.base_path}\\05_testing_validation\\unit_tests\\"',
                f"Move test files: {pattern}"
            )
        
        # GPU and monitoring tools
        tool_files = [
            "gpu_*.py", "*monitor*.py", "*benchmark*.py", "demo_multi_gpu.py"
        ]
        for pattern in tool_files:
            self.add_command(
                f'for %f in ("{self.base_path}\\{pattern}") do move "%f" "{self.base_path}\\06_tools_utilities\\gpu_tools\\"',
                f"Move tools: {pattern}"
            )
        
        # Model management
        model_files = [
            "download_*.py", "model_*.py", "find_models.py", "organize_models.py"
        ]
        for pattern in model_files:
            self.add_command(
                f'for %f in ("{self.base_path}\\{pattern}") do move "%f" "{self.base_path}\\03_models_storage\\model_configs\\"',
                f"Move model files: {pattern}"
            )
        
        # Batch files
        batch_files = ["*.bat"]
        for pattern in batch_files:
            self.add_command(
                f'for %f in ("{self.base_path}\\{pattern}") do move "%f" "{self.base_path}\\10_deployment\\batch_files\\"',
                f"Move batch files: {pattern}"
            )
    
    def generate_cleanup_commands(self):
        """Generate commands to clean up cache and temporary files"""
        print("🧹 GENERATING CLEANUP COMMANDS...")
        
        # Remove cache files
        cache_patterns = [
            "__pycache__", "*.pyc", "*.pyo", ".cache"
        ]
        
        for pattern in cache_patterns:
            if pattern.startswith("__") or pattern.startswith("."):
                # Remove directories
                self.add_command(
                    f'for /d /r "{self.base_path}" %d in ({pattern}) do @if exist "%d" rmdir /s /q "%d"',
                    f"Remove cache directories: {pattern}"
                )
            else:
                # Remove files
                self.add_command(
                    f'for /r "{self.base_path}" %f in ({pattern}) do @if exist "%f" del /q "%f"',
                    f"Remove cache files: {pattern}"
                )
        
        # Remove temporary/duplicate files
        temp_files = [
            "*_old.py", "*_simple.py", "simple_chat_old.py"
        ]
        for pattern in temp_files:
            self.add_command(
                f'for %f in ("{self.base_path}\\{pattern}") do del /q "%f"',
                f"Remove temporary file: {pattern}"
            )
        
        # Remove version control artifacts that shouldn't be in main directory
        version_files = ["1.20.0", "2.11.3", "25.1.1", "3.7", "4.5.0", "5.6.1"]
        for file in version_files:
            self.add_command(
                f'del /q "{self.base_path}\\{file}"',
                f"Remove version artifact: {file}"
            )
    
    def generate_consolidation_commands(self):
        """Generate commands to consolidate requirements"""
        print("📋 GENERATING CONSOLIDATION COMMANDS...")
        
        # Create unified requirements
        consolidation_script = f"""
echo # EZIO UNIFIED REQUIREMENTS - Generated {self.timestamp} > "{self.base_path}\\04_configuration\\requirements_files\\requirements_unified_final.txt"
echo # Consolidated from multiple requirement files >> "{self.base_path}\\04_configuration\\requirements_files\\requirements_unified_final.txt"
echo. >> "{self.base_path}\\04_configuration\\requirements_files\\requirements_unified_final.txt"
type "{self.base_path}\\04_configuration\\requirements_files\\requirements.txt" >> "{self.base_path}\\04_configuration\\requirements_files\\requirements_unified_final.txt" 2>nul
type "{self.base_path}\\04_configuration\\requirements_files\\requirements-unified.txt" >> "{self.base_path}\\04_configuration\\requirements_files\\requirements_unified_final.txt" 2>nul
type "{self.base_path}\\04_configuration\\requirements_files\\requirements_working.txt" >> "{self.base_path}\\04_configuration\\requirements_files\\requirements_unified_final.txt" 2>nul
"""
        self.add_command(consolidation_script, "Consolidate requirements files")
    
    def generate_documentation_commands(self):
        """Generate commands to organize documentation"""
        print("📊 GENERATING DOCUMENTATION COMMANDS...")
        
        # Move documentation files
        doc_files = [
            "README.md", "relatorio_desenvolvimento.md", "*.md"
        ]
        for pattern in doc_files:
            if "*" in pattern:
                self.add_command(
                    f'for %f in ("{self.base_path}\\{pattern}") do move "%f" "{self.base_path}\\07_documentation\\guides\\"',
                    f"Move documentation: {pattern}"
                )
            else:
                self.add_command(
                    f'move "{self.base_path}\\{pattern}" "{self.base_path}\\07_documentation\\guides\\"',
                    f"Move documentation: {pattern}"
                )
        
        # Move existing folders to appropriate locations
        folder_moves = [
            ("autogen_examples", "08_examples_demos\\autogen_examples"),
            ("autogen_generated", "08_examples_demos\\working_demos"),
            ("data", "09_data_cache\\shared_cache"),
            ("results", "05_testing_validation\\test_results"),
            ("reports", "07_documentation\\reports"),
            ("models", "03_models_storage\\local_models"),
            ("core", "01_core_system\\base_classes"),
            ("experts", "02_experts_modules\\financial_experts"),
            ("tools", "06_tools_utilities\\monitoring")
        ]
        
        for src, dst in folder_moves:
            self.add_command(
                f'xcopy /E /I /Y "{self.base_path}\\{src}" "{self.base_path}\\{dst}\\"',
                f"Move folder: {src} -> {dst}"
            )
            self.add_command(
                f'rmdir /s /q "{self.base_path}\\{src}"',
                f"Remove original folder: {src}"
            )
    
    def generate_final_report_command(self):
        """Generate command to create final report"""
        print("📋 GENERATING FINAL REPORT...")
        
        report_content = f"""
echo EZIO PROJECT ORGANIZATION REPORT > "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
echo ================================== >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
echo Timestamp: {self.timestamp} >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
echo Total Commands Executed: {len(self.commands)} >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
echo. >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
echo NEW STRUCTURE CREATED: >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
"""
        
        for directory in self.report["structure_created"]:
            report_content += f'echo   - {directory} >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"\n'
        
        report_content += f"""
echo. >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
echo ORGANIZATION COMPLETE! >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
echo Project is now organized and ready for development. >> "{self.base_path}\\07_documentation\\reports\\organization_report_{self.timestamp}.txt"
"""
        
        self.add_command(report_content, "Generate final organization report")
    
    def generate_all_commands(self):
        """Generate all automation commands"""
        print("🚀 EZIO PROJECT ORGANIZER CLI")
        print("=" * 50)
        print(f"📁 Base Path: {self.base_path}")
        print(f"🕐 Timestamp: {self.timestamp}")
        print("=" * 50)
        
        # Generate all command categories
        self.generate_new_structure()
        self.generate_file_organization_commands()
        self.generate_cleanup_commands()
        self.generate_consolidation_commands()
        self.generate_documentation_commands()
        self.generate_final_report_command()
        
        return self.commands
    
    def export_commands(self, format_type="batch"):
        """Export commands in specified format"""
        commands = self.generate_all_commands()
        
        if format_type == "batch":
            # Generate batch file
            batch_content = f"""@echo off
REM EZIO PROJECT ORGANIZER - AUTO-GENERATED {self.timestamp}
REM Total Commands: {len(commands)}
echo Starting EZIO Project Organization...

"""
            for i, cmd_obj in enumerate(commands, 1):
                batch_content += f"REM Step {i}: {cmd_obj['description']}\n"
                batch_content += f"{cmd_obj['command']}\n\n"
            
            batch_content += """
echo.
echo ========================================
echo EZIO PROJECT ORGANIZATION COMPLETE!
echo ========================================
echo Project structure has been reorganized.
echo Check the reports folder for detailed log.
pause
"""
            return batch_content
        
        elif format_type == "copilot":
            # Generate copilot-friendly commands
            copilot_content = f"""# EZIO PROJECT ORGANIZER - COPILOT COMMANDS
# Generated: {self.timestamp}
# Total Commands: {len(commands)}

"""
            for i, cmd_obj in enumerate(commands, 1):
                copilot_content += f"# Step {i}: {cmd_obj['description']}\n"
                copilot_content += f"{cmd_obj['command']}\n\n"
            
            return copilot_content
        
        elif format_type == "json":
            # Generate JSON format for programmatic use
            return json.dumps({
                "metadata": {
                    "timestamp": self.timestamp,
                    "total_commands": len(commands),
                    "base_path": self.base_path
                },
                "commands": commands,
                "report": self.report
            }, indent=2)

def main():
    """Main execution function"""
    organizer = EzioOrganizerCLI()
    
    print("🎯 CHOOSE OUTPUT FORMAT:")
    print("1. Batch File (.bat) - Ready to execute")
    print("2. Copilot Commands - Copy/paste friendly")
    print("3. JSON Export - Programmatic format")
    print()
    
    # For this demo, we'll generate all formats
    print("🚀 GENERATING ALL FORMATS...")
    print()
    
    # Generate batch file
    batch_content = organizer.export_commands("batch")
    batch_file = f"ezio_organizer_{organizer.timestamp}.bat"
    with open(batch_file, 'w', encoding='utf-8') as f:
        f.write(batch_content)
    print(f"✅ Batch file created: {batch_file}")
    
    # Generate copilot commands
    copilot_content = organizer.export_commands("copilot")
    copilot_file = f"ezio_copilot_commands_{organizer.timestamp}.txt"
    with open(copilot_file, 'w', encoding='utf-8') as f:
        f.write(copilot_content)
    print(f"✅ Copilot commands created: {copilot_file}")
    
    # Generate JSON export
    json_content = organizer.export_commands("json")
    json_file = f"ezio_organization_plan_{organizer.timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(json_content)
    print(f"✅ JSON export created: {json_file}")
    
    print()
    print("🎉 ALL FILES GENERATED!")
    print("🤖 FOR COPILOT: Copy/paste commands from the .txt file")
    print("⚡ FOR IMMEDIATE: Run the .bat file")
    print("🔧 FOR CUSTOM: Use the .json file")
    print()
    print(f"📊 Total Commands Generated: {len(organizer.commands)}")
    print("🚀 Ready for execution!")

if __name__ == "__main__":
    main()