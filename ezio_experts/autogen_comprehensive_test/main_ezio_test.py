#!/usr/bin/env python3
"""
EzioFilho Unified AutoGen Comprehensive Test Suite
Purpose: Complete system validation and AutoGen integration testing
Author: marcobarreto007
Date: 2025-05-24 02:51:56 UTC
System: Windows 10.0.19045.5854
"""

import os
import sys
import json
import logging
import traceback
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# REM System Configuration
PROJECT_ROOT = r"C:\Users\anapa\eziofilho-unified"
USER_NAME = "marcobarreto007"
TEST_TIMESTAMP = "2025-05-24 02:51:56 UTC"
WINDOWS_VERSION = "10.0.19045.5854"

class EzioSystemTester:
    """Comprehensive test suite for EzioFilho Unified System"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(PROJECT_ROOT)
        self.test_results = {}
        self.start_time = datetime.now()
        self.passed_tests = 0
        self.total_tests = 0
        
        # REM Add project to Python path if it exists
        if self.project_root.exists():
            sys.path.insert(0, str(self.project_root))
            self.logger.info(f"✅ Added to Python path: {self.project_root}")
    
    def setup_logging(self):
        """Configure comprehensive logging system"""
        log_format = '%(asctime)s | %(levelname)8s | %(funcName)s | %(message)s'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('ezio_test_complete.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # REM Suppress verbose third-party loggers
        for logger_name in ['urllib3', 'requests', 'matplotlib']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def print_system_banner(self):
        """Print comprehensive system information banner"""
        banner = f"""
{'='*80}
🚀 EZIOFILHO UNIFIED SYSTEM - COMPREHENSIVE TEST SUITE
{'='*80}
👤 User: {USER_NAME}
📅 Test Date: {TEST_TIMESTAMP}
🖥️  Windows: {WINDOWS_VERSION}
🐍 Python: {sys.version.split()[0]} ({sys.platform})
📁 Project Root: {self.project_root}
⏰ Test Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        print(banner)
        self.logger.info("Test suite initialized")
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Execute a single test with error handling"""
        self.total_tests += 1
        
        try:
            self.logger.info(f"🧪 Running: {test_name}")
            result = test_func()
            
            if result:
                self.passed_tests += 1
                self.logger.info(f"✅ PASSED: {test_name}")
            else:
                self.logger.error(f"❌ FAILED: {test_name}")
                
            self.test_results[test_name] = {
                "status": "PASSED" if result else "FAILED",
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"💥 EXCEPTION in {test_name}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            self.test_results[test_name] = {
                "status": "EXCEPTION",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            return False
    
    def test_python_environment(self) -> bool:
        """Test Python environment and basic modules"""
        self.logger.info("=== PYTHON ENVIRONMENT TEST ===")
        
        # REM Test Python version
        version_info = sys.version_info
        if version_info.major >= 3 and version_info.minor >= 7:
            self.logger.info(f"✅ Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
        else:
            self.logger.error(f"❌ Python version too old: {version_info}")
            return False
        
        # REM Test essential modules
        essential_modules = ['json', 'logging', 'pathlib', 'datetime', 'subprocess']
        
        for module in essential_modules:
            try:
                __import__(module)
                self.logger.info(f"✅ Module available: {module}")
            except ImportError:
                self.logger.error(f"❌ Module missing: {module}")
                return False
        
        # REM Test pip availability
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info(f"✅ Pip available: {result.stdout.strip()}")
            else:
                self.logger.warning("⚠️  Pip not available")
        except Exception as e:
            self.logger.warning(f"⚠️  Pip check failed: {e}")
        
        return True
    
    def test_project_structure(self) -> bool:
        """Test EzioFilho project directory structure"""
        self.logger.info("=== PROJECT STRUCTURE TEST ===")
        
        if not self.project_root.exists():
            self.logger.error(f"❌ Project root not found: {self.project_root}")
            return False
        
        # REM Required directories
        required_dirs = [
            "01_core_system",
            "02_experts_modules", 
            "03_models_storage",
            "04_configuration",
            "05_testing_validation",
            "06_tools_utilities",
            "07_documentation",
            "08_examples_demos",
            "09_data_cache",
            "10_deployment"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.logger.info(f"✅ Directory: {dir_name}")
            else:
                self.logger.warning(f"⚠️  Missing: {dir_name}")
                missing_dirs.append(dir_name)
        
        # REM Optional but important directories
        optional_dirs = [
            "ezio_experts",
            "autogen_generated", 
            "testes_autogen",
            "testes_experts",
            "tests_modelos"
        ]
        
        for dir_name in optional_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.logger.info(f"✅ Optional: {dir_name}")
            else:
                self.logger.info(f"ℹ️  Optional missing: {dir_name}")
        
        return len(missing_dirs) < 3  # REM Allow some missing dirs
    
    def test_key_files(self) -> bool:
        """Test presence and basic validation of key files"""
        self.logger.info("=== KEY FILES TEST ===")
        
        key_files = {
            "ezio_organizer_cli.py": {"required": True, "min_size": 1000},
            "model_auto_discovery.py": {"required": True, "min_size": 1000},
            "test_hf_api.py": {"required": False, "min_size": 100},
            "README.md": {"required": False, "min_size": 100},
            "model_inventory.txt": {"required": False, "min_size": 0}
        }
        
        missing_required = []
        
        for filename, config in key_files.items():
            file_path = self.project_root / filename
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                
                if file_size >= config["min_size"]:
                    self.logger.info(f"✅ File: {filename} ({file_size} bytes)")
                else:
                    self.logger.warning(f"⚠️  File too small: {filename} ({file_size} bytes)")
                    
            else:
                if config["required"]:
                    self.logger.error(f"❌ Required file missing: {filename}")
                    missing_required.append(filename)
                else:
                    self.logger.info(f"ℹ️  Optional file missing: {filename}")
        
        return len(missing_required) == 0
    
    def test_cache_system(self) -> bool:
        """Test cache directories and structure"""
        self.logger.info("=== CACHE SYSTEM TEST ===")
        
        cache_base = Path(r"C:\Users\anapa\.cache")
        
        if not cache_base.exists():
            self.logger.warning("⚠️  Main cache directory not found")
            return False
        
        # REM Check specific cache directories
        cache_dirs = ["eziofilho", "huggingface", "models"]
        found_caches = 0
        
        for cache_dir in cache_dirs:
            cache_path = cache_base / cache_dir
            if cache_path.exists():
                self.logger.info(f"✅ Cache: {cache_dir}")
                found_caches += 1
            else:
                self.logger.info(f"ℹ️  Cache not found: {cache_dir}")
        
        self.logger.info(f"✅ Cache system: {found_caches}/{len(cache_dirs)} directories found")
        return found_caches >= 1  # REM At least one cache dir should exist
    
    def test_autogen_integration(self) -> bool:
        """Test AutoGen availability and integration"""
        self.logger.info("=== AUTOGEN INTEGRATION TEST ===")
        
        try:
            # REM Try to import autogen
            import autogen
            self.logger.info(f"✅ AutoGen imported successfully")
            
            try:
                version = getattr(autogen, '__version__', 'unknown')
                self.logger.info(f"✅ AutoGen version: {version}")
            except:
                self.logger.info("ℹ️  AutoGen version unknown")
            
            # REM Check AutoGen directories
            autogen_dirs = ["autogen_generated", "testes_autogen"]
            
            for dir_name in autogen_dirs:
                dir_path = self.project_root / dir_name
                if dir_path.exists():
                    file_count = len(list(dir_path.glob("*")))
                    self.logger.info(f"✅ AutoGen dir: {dir_name} ({file_count} items)")
                else:
                    self.logger.info(f"ℹ️  AutoGen dir missing: {dir_name}")
            
            return True
            
        except ImportError:
            self.logger.warning("⚠️  AutoGen not installed")
            
            # REM Try to install AutoGen
            self.logger.info("🔄 Attempting to install AutoGen...")
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'pyautogen'
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    self.logger.info("✅ AutoGen installation successful")
                    return True
                else:
                    self.logger.error(f"❌ AutoGen installation failed: {result.stderr}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ AutoGen installation exception: {e}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ AutoGen test exception: {e}")
            return False
    
    def test_expert_modules(self) -> bool:
        """Test expert modules and components"""
        self.logger.info("=== EXPERT MODULES TEST ===")
        
        # REM Check experts directory
        experts_dir = self.project_root / "ezio_experts"
        
        if not experts_dir.exists():
            self.logger.warning("⚠️  Experts directory not found")
            return False
        
        # REM List expert modules
        expert_items = list(experts_dir.glob("*"))
        expert_dirs = [item for item in expert_items if item.is_dir()]
        expert_files = [item for item in expert_items if item.is_file() and item.suffix == '.py']
        
        self.logger.info(f"✅ Expert directories: {len(expert_dirs)}")
        for expert_dir in expert_dirs:
            self.logger.info(f"   📁 {expert_dir.name}")
        
        self.logger.info(f"✅ Expert Python files: {len(expert_files)}")
        for expert_file in expert_files:
            self.logger.info(f"   🐍 {expert_file.name}")
        
        # REM Check specific expert types
        expert_types = ["experts_fingpt"]
        for expert_type in expert_types:
            expert_path = self.project_root / expert_type
            if expert_path.exists():
                self.logger.info(f"✅ Specialized expert: {expert_type}")
            else:
                self.logger.info(f"ℹ️  Specialized expert missing: {expert_type}")
        
        return len(expert_dirs) + len(expert_files) > 0
    
    def test_model_system(self) -> bool:
        """Test model discovery and management system"""
        self.logger.info("=== MODEL SYSTEM TEST ===")
        
        # REM Test model discovery script
        model_discovery = self.project_root / "model_auto_discovery.py"
        if model_discovery.exists():
            self.logger.info(f"✅ Model discovery script: {model_discovery.stat().st_size} bytes")
        else:
            self.logger.error("❌ Model discovery script not found")
            return False
        
        # REM Test model inventory
        model_inventory = self.project_root / "model_inventory.txt"
        if model_inventory.exists():
            with open(model_inventory, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                self.logger.info(f"✅ Model inventory: {len(lines)} entries")
        else:
            self.logger.info("ℹ️  Model inventory not found")
        
        # REM Test model download scripts
        download_scripts = ["download_models.py", "download_mistral.py"]
        
        for script_name in download_scripts:
            script_path = self.project_root / script_name
            if script_path.exists():
                self.logger.info(f"✅ Download script: {script_name}")
            else:
                self.logger.info(f"ℹ️  Download script missing: {script_name}")
        
        # REM Test model storage directory
        model_storage = self.project_root / "03_models_storage"
        if model_storage.exists():
            model_count = len(list(model_storage.glob("**/*")))
            self.logger.info(f"✅ Model storage: {model_count} items")
        else:
            self.logger.warning("⚠️  Model storage directory not found")
        
        return True
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "test_metadata": {
                "user": USER_NAME,
                "test_date": TEST_TIMESTAMP,
                "windows_version": WINDOWS_VERSION,
                "python_version": sys.version,
                "project_root": str(self.project_root),
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds()
            },
            "test_summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.total_tests - self.passed_tests,
                "success_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [name for name, result in self.test_results.items() 
                       if result["status"] != "PASSED"]
        
        if "AutoGen Integration" in failed_tests:
            recommendations.append("Install AutoGen: pip install pyautogen")
        
        if "Project Structure" in failed_tests:
            recommendations.append("Create missing project directories")
        
        if "Key Files" in failed_tests:
            recommendations.append("Ensure critical Python files are present")
        
        if not failed_tests:
            recommendations.append("System is ready for full AutoGen integration!")
            recommendations.append("Consider running specific expert module tests")
            recommendations.append("Test model download and discovery functionality")
        
        return recommendations
    
    def run_comprehensive_test_suite(self):
        """Execute the complete test suite"""
        self.print_system_banner()
        
        # REM Define test sequence
        test_sequence = [
            ("Python Environment", self.test_python_environment),
            ("Project Structure", self.test_project_structure),
            ("Key Files", self.test_key_files),
            ("Cache System", self.test_cache_system),
            ("AutoGen Integration", self.test_autogen_integration),
            ("Expert Modules", self.test_expert_modules),
            ("Model System", self.test_model_system)
        ]
        
        # REM Execute all tests
        self.logger.info(f"🚀 Starting {len(test_sequence)} tests...")
        
        for test_name, test_func in test_sequence:
            self.run_test(test_name, test_func)
            print("-" * 60)  # REM Visual separator
        
        # REM Generate and save report
        report = self.generate_comprehensive_report()
        
        try:
            with open('ezio_test_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info("💾 Test report saved: ezio_test_report.json")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save report: {e}")
        
        # REM Print final summary
        self.print_final_summary(report)
        
        return report
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print comprehensive final summary"""
        summary = report["test_summary"]
        recommendations = report["recommendations"]
        
        print("\n" + "="*80)
        print("🏁 EZIOFILHO UNIFIED TEST SUITE - FINAL SUMMARY")
        print("="*80)
        print(f"📊 Tests Executed: {summary['total_tests']}")
        print(f"✅ Tests Passed: {summary['passed_tests']}")
        print(f"❌ Tests Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Duration: {report['test_metadata']['duration_seconds']:.2f} seconds")
        
        print("\n🔍 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"   {status_icon} {test_name}: {result['status']}")
        
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        
        if summary['success_rate'] >= 80:
            print("🎉 SYSTEM STATUS: READY FOR PRODUCTION")
            self.logger.info("System passed comprehensive testing")
        elif summary['success_rate'] >= 60:
            print("⚠️  SYSTEM STATUS: NEEDS MINOR FIXES")
            self.logger.warning("System needs minor improvements")
        else:
            print("🚨 SYSTEM STATUS: REQUIRES MAJOR ATTENTION")
            self.logger.error("System needs significant improvements")
        
        print("="*80)

def main():
    """Main execution function"""
    try:
        # REM Initialize and run comprehensive test
        tester = EzioSystemTester()
        report = tester.run_comprehensive_test_suite()
        
        # REM Exit with appropriate code
        success_rate = report["test_summary"]["success_rate"]
        
        if success_rate >= 80:
            sys.exit(0)  # REM Success
        elif success_rate >= 60:
            sys.exit(1)  # REM Warning
        else:
            sys.exit(2)  # REM Error
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Critical test failure: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()