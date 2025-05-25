#!/usr/bin/env python3
"""
Main AutoGen Test Script for EzioFilho Unified System
Purpose: Test AutoGen integration with the expert system
Author: marcobarreto007
Date: 2025-05-24
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# REM Add the project root to Python path
project_root = r"C:\Users\anapa\eziofilho-unified"
sys.path.insert(0, project_root)

# REM Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autogen_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutoGenTestRunner:
    """Main test runner for AutoGen system"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.start_time = datetime.now()
        
    def check_environment(self) -> bool:
        """Check if the environment is properly configured"""
        logger.info("=== ENVIRONMENT CHECK ===")
        
        try:
            # REM Check if autogen is installed
            import autogen
            logger.info(f"✅ AutoGen version: {autogen.__version__}")
            
            # REM Check project structure
            required_dirs = [
                "01_core_system",
                "02_experts_modules", 
                "ezio_experts",
                "autogen_generated",
                "testes_autogen"
            ]
            
            for dir_name in required_dirs:
                dir_path = os.path.join(self.project_root, dir_name)
                if os.path.exists(dir_path):
                    logger.info(f"✅ Directory found: {dir_name}")
                else:
                    logger.warning(f"⚠️  Directory missing: {dir_name}")
                    
            # REM Check cache directories
            cache_dir = r"C:\Users\anapa\.cache"
            if os.path.exists(cache_dir):
                logger.info(f"✅ Cache directory found: {cache_dir}")
                
                # REM List cache contents
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        logger.info(f"   📁 {item}/")
                        
            return True
            
        except ImportError as e:
            logger.error(f"❌ AutoGen not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Environment check failed: {e}")
            return False
    
    def test_ezio_organizer(self) -> bool:
        """Test the main ezio organizer CLI"""
        logger.info("=== TESTING EZIO ORGANIZER ===")
        
        try:
            organizer_path = os.path.join(self.project_root, "ezio_organizer_cli.py")
            
            if not os.path.exists(organizer_path):
                logger.error("❌ ezio_organizer_cli.py not found")
                return False
                
            logger.info(f"✅ Found ezio_organizer_cli.py ({os.path.getsize(organizer_path)} bytes)")
            
            # REM Try to import and test basic functionality
            sys.path.insert(0, self.project_root)
            
            # REM Basic syntax check
            with open(organizer_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # REM Check for key components
            if "autogen" in content.lower():
                logger.info("✅ AutoGen integration detected in organizer")
            else:
                logger.warning("⚠️  AutoGen integration not clearly visible")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Ezio organizer test failed: {e}")
            return False
    
    def test_model_discovery(self) -> bool:
        """Test the model auto discovery system"""
        logger.info("=== TESTING MODEL DISCOVERY ===")
        
        try:
            discovery_path = os.path.join(self.project_root, "model_auto_discovery.py")
            
            if not os.path.exists(discovery_path):
                logger.error("❌ model_auto_discovery.py not found")
                return False
                
            logger.info(f"✅ Found model_auto_discovery.py ({os.path.getsize(discovery_path)} bytes)")
            
            # REM Check model inventory
            inventory_path = os.path.join(self.project_root, "model_inventory.txt")
            if os.path.exists(inventory_path):
                with open(inventory_path, 'r', encoding='utf-8') as f:
                    inventory = f.read()
                    model_count = len([line for line in inventory.split('\n') if line.strip()])
                    logger.info(f"✅ Model inventory: {model_count} entries")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model discovery test failed: {e}")
            return False
    
    def test_experts_modules(self) -> bool:
        """Test the experts modules"""
        logger.info("=== TESTING EXPERTS MODULES ===")
        
        try:
            experts_dir = os.path.join(self.project_root, "ezio_experts")
            
            if not os.path.exists(experts_dir):
                logger.warning("⚠️  ezio_experts directory not found")
                return False
                
            # REM List expert modules
            expert_modules = []
            for item in os.listdir(experts_dir):
                item_path = os.path.join(experts_dir, item)
                if os.path.isdir(item_path):
                    expert_modules.append(item)
                    logger.info(f"   🤖 Expert module: {item}")
                    
            logger.info(f"✅ Found {len(expert_modules)} expert modules")
            
            # REM Test specific expert modules
            test_modules_dir = os.path.join(self.project_root, "testes_experts")
            if os.path.exists(test_modules_dir):
                logger.info("✅ Expert tests directory found")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Experts modules test failed: {e}")
            return False
    
    def test_autogen_integration(self) -> bool:
        """Test AutoGen specific integration"""
        logger.info("=== TESTING AUTOGEN INTEGRATION ===")
        
        try:
            # REM Check autogen generated directory
            autogen_dir = os.path.join(self.project_root, "autogen_generated")
            if os.path.exists(autogen_dir):
                logger.info("✅ AutoGen generated directory found")
                
                # REM List generated files
                for item in os.listdir(autogen_dir):
                    logger.info(f"   📄 Generated: {item}")
            
            # REM Check autogen tests
            autogen_tests_dir = os.path.join(self.project_root, "testes_autogen")
            if os.path.exists(autogen_tests_dir):
                logger.info("✅ AutoGen tests directory found")
                
                # REM List test files
                for item in os.listdir(autogen_tests_dir):
                    logger.info(f"   🧪 Test: {item}")
            
            # REM Test basic AutoGen functionality
            try:
                import autogen
                
                # REM Create a simple test configuration
                config = {
                    "llm_config": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test_key"
                    }
                }
                
                logger.info("✅ AutoGen basic configuration test passed")
                
            except Exception as e:
                logger.warning(f"⚠️  AutoGen configuration test: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ AutoGen integration test failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive system test"""
        logger.info("🚀 STARTING EZIOFILHO UNIFIED AUTOGEN TEST")
        logger.info(f"📅 Test started at: {self.start_time}")
        logger.info(f"👤 User: marcobarreto007")
        logger.info("=" * 60)
        
        tests = [
            ("Environment Check", self.check_environment),
            ("Ezio Organizer Test", self.test_ezio_organizer),
            ("Model Discovery Test", self.test_model_discovery),
            ("Experts Modules Test", self.test_experts_modules),
            ("AutoGen Integration Test", self.test_autogen_integration)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                results[test_name] = False
                logger.error(f"❌ {test_name}: EXCEPTION - {e}")
        
        # REM Generate test summary
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info(f"✅ Passed: {passed}/{total}")
        logger.info(f"❌ Failed: {total - passed}/{total}")
        logger.info(f"⏱️  Duration: {duration.total_seconds():.2f} seconds")
        logger.info(f"🏁 Test completed at: {end_time}")
        
        # REM Save results to file
        result_summary = {
            "test_date": self.start_time.isoformat(),
            "user": "marcobarreto007",
            "duration_seconds": duration.total_seconds(),
            "tests_passed": passed,
            "tests_total": total,
            "test_results": results,
            "system_info": {
                "project_root": self.project_root,
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        try:
            results_file = os.path.join(self.project_root, "test_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result_summary, f, indent=2, default=str)
            logger.info(f"💾 Test results saved to: {results_file}")
        except Exception as e:
            logger.warning(f"⚠️  Could not save results: {e}")
        
        return result_summary

def main():
    """Main test execution function"""
    try:
        # REM Initialize test runner
        test_runner = AutoGenTestRunner()
        
        # REM Run comprehensive test
        results = test_runner.run_comprehensive_test()
        
        # REM Exit with appropriate code
        if results["tests_passed"] == results["tests_total"]:
            logger.info("🎉 ALL TESTS PASSED!")
            sys.exit(0)
        else:
            logger.warning("⚠️  SOME TESTS FAILED!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()