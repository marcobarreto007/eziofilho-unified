# enhanced_trading_system.py
"""
EZIO FINANCIAL AI - Enhanced Trading System
Integration with Existing Expert Modules
Author: Sistema Ezio
Date: 2025-05-25
Version: 2.0 - Full Integration
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add existing modules to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "02_experts_modules"))

# Import existing systems
try:
    from advanced_model_manager import AdvancedModelManager
    print("✅ Advanced Model Manager imported")
except ImportError as e:
    print(f"⚠️ Advanced Model Manager not available: {e}")
    AdvancedModelManager = None

try:
    from expert_fingpt import FinGPTExpert
    print("✅ FinGPT Expert imported")
except ImportError as e:
    print(f"⚠️ FinGPT Expert not available: {e}")
    FinGPTExpert = None

class EnhancedTradingSystem:
    """Enhanced trading system integrating existing modules"""
    
    def __init__(self):
        self.base_path = Path("C:/Users/anapa/eziofilho-unified")
        self.experts = {}
        self.model_manager = None
        
        print("🚀 EZIO ENHANCED TRADING SYSTEM")
        print("=" * 50)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"📁 Base: {self.base_path}")
        print("=" * 50)
        
        self._setup_logging()
        self._initialize_model_manager()
        self._register_existing_experts()
    
    def _setup_logging(self):
        """Setup logging system"""
        log_dir = self.base_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"enhanced_system_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        
        logging.info("Enhanced trading system initialized")
    
    def _initialize_model_manager(self):
        """Initialize advanced model manager"""
        if AdvancedModelManager:
            try:
                self.model_manager = AdvancedModelManager()
                print("✅ Advanced Model Manager initialized")
                logging.info("Model manager initialized successfully")
            except Exception as e:
                print(f"❌ Model manager initialization failed: {e}")
                logging.error(f"Model manager failed: {e}")
        else:
            print("⚠️ Advanced Model Manager not available")
    
    def _register_existing_experts(self):
        """Register existing expert systems"""
        print("\n🧠 REGISTERING EXISTING EXPERTS:")
        
        # Register FinGPT Expert
        if FinGPTExpert:
            try:
                self.experts['fingpt'] = FinGPTExpert()
                print("✅ FinGPT Expert registered")
                logging.info("FinGPT expert registered")
            except Exception as e:
                print(f"❌ FinGPT registration failed: {e}")
                logging.error(f"FinGPT registration failed: {e}")
        
        # Scan for additional experts in experts directory
        experts_dir = self.base_path / "02_experts_modules" / "experts"
        if experts_dir.exists():
            print(f"🔍 Scanning experts directory: {experts_dir}")
            for expert_file in experts_dir.glob("*.py"):
                if expert_file.name != "__init__.py":
                    print(f"📄 Found expert module: {expert_file.name}")
        
        print(f"📊 Total experts registered: {len(self.experts)}")
    
    def run_system_analysis(self):
        """Run comprehensive system analysis"""
        print("\n📊 SYSTEM ANALYSIS:")
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "existing_modules": self._analyze_existing_modules(),
            "model_status": self._check_model_status(),
            "expert_status": self._check_experts(),
            "integration_status": "active"
        }
        
        # Display analysis
        modules = analysis["existing_modules"]
        print(f"📁 Existing modules: {modules['count']} found")
        for module in modules['modules']:
            print(f"  - {module['name']}: {module['size']}KB")
        
        if analysis["model_status"]["available"]:
            print(f"🤖 Model Manager: ✅ {analysis['model_status']['type']}")
        else:
            print("🤖 Model Manager: ❌ Not available")
        
        print(f"👥 Active experts: {len(self.experts)}")
        
        # Save analysis
        reports_dir = self.base_path / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"system_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n📄 Analysis saved: {report_file}")
        return analysis
    
    def _analyze_existing_modules(self):
        """Analyze existing module structure"""
        modules_dir = self.base_path / "02_experts_modules"
        modules = []
        
        if modules_dir.exists():
            for py_file in modules_dir.glob("*.py"):
                size_kb = py_file.stat().st_size / 1024
                modules.append({
                    "name": py_file.name,
                    "size": round(size_kb, 1),
                    "path": str(py_file)
                })
        
        return {
            "count": len(modules),
            "modules": modules,
            "total_size_kb": sum(m["size"] for m in modules)
        }
    
    def _check_model_status(self):
        """Check model manager status"""
        if self.model_manager:
            return {
                "available": True,
                "type": "AdvancedModelManager",
                "status": "initialized"
            }
        else:
            return {
                "available": False,
                "type": None,
                "status": "not_available"
            }
    
    def _check_experts(self):
        """Check expert status"""
        return {
            "registered": len(self.experts),
            "experts": list(self.experts.keys()),
            "status": "active" if self.experts else "none"
        }
    
    def test_expert_integration(self):
        """Test expert integration"""
        print("\n🧪 TESTING EXPERT INTEGRATION:")
        
        test_results = {}
        
        for expert_name, expert in self.experts.items():
            try:
                # Test basic functionality
                if hasattr(expert, 'analyze') and callable(expert.analyze):
                    result = expert.analyze({"test": True, "symbol": "AAPL"})
                    test_results[expert_name] = {
                        "status": "success",
                        "response": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    }
                    print(f"✅ {expert_name}: Working")
                else:
                    test_results[expert_name] = {
                        "status": "no_analyze_method",
                        "response": "Expert lacks analyze method"
                    }
                    print(f"⚠️ {expert_name}: No analyze method")
                    
            except Exception as e:
                test_results[expert_name] = {
                    "status": "error",
                    "response": str(e)
                }
                print(f"❌ {expert_name}: {e}")
        
        return test_results
    
    def generate_expansion_plan(self):
        """Generate plan for 12-expert system"""
        print("\n🎯 12-EXPERT EXPANSION PLAN:")
        
        target_experts = [
            "market_analyst", "technical_analyst", "fundamental_analyst",
            "risk_manager", "portfolio_optimizer", "sentiment_analyzer",
            "news_processor", "economic_indicator", "options_strategist",
            "crypto_specialist", "forex_expert", "commodities_analyst"
        ]
        
        existing_experts = list(self.experts.keys())
        needed_experts = [exp for exp in target_experts if exp not in existing_experts]
        
        print(f"📊 Target experts: {len(target_experts)}")
        print(f"✅ Existing: {len(existing_experts)} - {existing_experts}")
        print(f"🔧 Needed: {len(needed_experts)} - {needed_experts}")
        
        expansion_plan = {
            "total_target": len(target_experts),
            "existing_count": len(existing_experts),
            "needed_count": len(needed_experts),
            "existing_experts": existing_experts,
            "needed_experts": needed_experts,
            "completion_percentage": (len(existing_experts) / len(target_experts)) * 100
        }
        
        print(f"🎯 Completion: {expansion_plan['completion_percentage']:.1f}%")
        
        return expansion_plan

def main():
    """Main execution function"""
    try:
        # Initialize enhanced system
        system = EnhancedTradingSystem()
        
        # Run system analysis
        analysis = system.run_system_analysis()
        
        # Test expert integration
        test_results = system.test_expert_integration()
        
        # Generate expansion plan
        expansion_plan = system.generate_expansion_plan()
        
        print("\n" + "=" * 50)
        print("✅ ENHANCED SYSTEM ANALYSIS COMPLETE!")
        print(f"🎯 Ready for {expansion_plan['needed_count']} additional experts")
        print("=" * 50)
        
        return system
        
    except Exception as e:
        print(f"\n❌ ENHANCED SYSTEM ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    system = main()
    if system:
        print("\n🚀 Enhanced system ready for expert expansion!")
    else:
        print("\n🚨 Enhanced system failed to initialize")