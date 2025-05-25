# system_compatibility_check.py - Complete system compatibility check
# Audit Mode: Active - System compatibility diagnostic
# Path: C:\Users\anapa\eziofilho-unified\ezio_experts\diagnostics
# User: marcobarreto007
# Date: 2025-05-24 21:14:04 UTC
# Objective: Diagnose system compatibility issues

import os
import sys
import platform
import subprocess
import json
from pathlib import Path
import ctypes

print("=" * 80)
print("🔍 SYSTEM COMPATIBILITY DIAGNOSTIC")
print("=" * 80)

class SystemDiagnostic:
    """Complete system diagnostic for AI models"""
    
    def __init__(self):
        self.issues = []
        self.solutions = []
        self.system_info = {}
        
    def check_windows_version(self):
        """Check Windows version and build"""
        print("\n📊 Windows Version Check:")
        
        try:
            # Get Windows version
            version = platform.version()
            release = platform.release()
            build = platform.win32_ver()[1]
            edition = platform.win32_edition()
            
            self.system_info['windows'] = {
                'version': version,
                'release': release,
                'build': build,
                'edition': edition
            }
            
            print(f"✅ Windows {release} (Build {build})")
            print(f"   Edition: {edition}")
            
            # Check if 64-bit
            is_64bit = platform.machine().endswith('64')
            print(f"   Architecture: {'64-bit' if is_64bit else '32-bit'}")
            
            if not is_64bit:
                self.issues.append("32-bit Windows detected")
                self.solutions.append("Upgrade to 64-bit Windows for AI models")
                
        except Exception as e:
            print(f"❌ Error checking Windows: {e}")
            
    def check_cpu_features(self):
        """Check CPU features required for AI"""
        print("\n🖥️ CPU Features Check:")
        
        try:
            # Check CPU info
            cpu_info = platform.processor()
            print(f"✅ CPU: {cpu_info}")
            
            # Check for AVX support (required for many AI models)
            try:
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'Name,NumberOfCores,NumberOfLogicalProcessors'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(result.stdout)
            except:
                pass
                
            # Check if AVX is supported
            try:
                import numpy as np
                # Try to use AVX instructions
                a = np.random.rand(100)
                b = np.random.rand(100)
                c = np.dot(a, b)
                print("✅ AVX instructions: Supported")
            except:
                self.issues.append("AVX instructions may not be supported")
                self.solutions.append("Some AI models require AVX support")
                
        except Exception as e:
            print(f"❌ Error checking CPU: {e}")
            
    def check_memory(self):
        """Check system memory"""
        print("\n💾 Memory Check:")
        
        try:
            # Get memory info using ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            
            meminfo = MEMORYSTATUSEX()
            meminfo.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(meminfo))
            
            total_ram = meminfo.ullTotalPhys / (1024**3)  # Convert to GB
            available_ram = meminfo.ullAvailPhys / (1024**3)
            
            print(f"✅ Total RAM: {total_ram:.1f} GB")
            print(f"   Available: {available_ram:.1f} GB")
            
            if total_ram < 8:
                self.issues.append(f"Low RAM: {total_ram:.1f} GB")
                self.solutions.append("Minimum 8GB RAM recommended for AI models")
                
        except Exception as e:
            print(f"❌ Error checking memory: {e}")
            
    def check_python_packages(self):
        """Check Python packages compatibility"""
        print("\n🐍 Python Packages Check:")
        
        packages = {
            'numpy': 'Core numerical computing',
            'torch': 'PyTorch for deep learning',
            'transformers': 'Hugging Face transformers',
            'langchain': 'LangChain framework',
            'autogen': 'AutoGen for multi-agent',
            'openai': 'OpenAI API client'
        }
        
        for package, description in packages.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package}: {version} - {description}")
            except ImportError:
                print(f"❌ {package}: Not installed - {description}")
                self.issues.append(f"{package} not installed")
                
    def check_visual_cpp_redist(self):
        """Check Visual C++ Redistributables"""
        print("\n🔧 Visual C++ Redistributables Check:")
        
        # Common paths for VC++ redist
        vc_paths = [
            Path("C:/Windows/System32/msvcp140.dll"),
            Path("C:/Windows/System32/vcruntime140.dll"),
            Path("C:/Windows/System32/vcruntime140_1.dll")
        ]
        
        missing = []
        for path in vc_paths:
            if path.exists():
                print(f"✅ {path.name}: Found")
            else:
                print(f"❌ {path.name}: Missing")
                missing.append(path.name)
                
        if missing:
            self.issues.append("Visual C++ Redistributables missing")
            self.solutions.append("Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            
    def check_gpu_drivers(self):
        """Check GPU and drivers"""
        print("\n🎮 GPU Check:")
        
        try:
            # Try nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ NVIDIA GPU: {result.stdout.strip()}")
            else:
                print("❌ No NVIDIA GPU detected")
        except:
            print("❌ nvidia-smi not found - No NVIDIA drivers")
            
    def generate_report(self):
        """Generate diagnostic report"""
        print("\n" + "="*80)
        print("📋 DIAGNOSTIC REPORT")
        print("="*80)
        
        if not self.issues:
            print("✅ No compatibility issues found!")
        else:
            print(f"⚠️  Found {len(self.issues)} issues:\n")
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
                if i <= len(self.solutions):
                    print(f"   💡 Solution: {self.solutions[i-1]}")
                    
        # Save report
        report = {
            'timestamp': platform.datetime.datetime.now().isoformat(),
            'system_info': self.system_info,
            'issues': self.issues,
            'solutions': self.solutions
        }
        
        report_path = Path("diagnostic_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Report saved: {report_path}")
        
    def fix_common_issues(self):
        """Attempt to fix common issues"""
        print("\n🔧 Attempting automatic fixes...")
        
        # Fix 1: Install missing Visual C++ components
        if any("Visual C++" in issue for issue in self.issues):
            print("\n1. Installing Visual C++ Build Tools...")
            cmd = f"{sys.executable} -m pip install --upgrade setuptools wheel"
            subprocess.run(cmd, shell=True)
            
        # Fix 2: Install/upgrade core packages
        print("\n2. Installing/upgrading core packages...")
        packages = ['numpy', 'wheel', 'setuptools', 'pip']
        for pkg in packages:
            cmd = f"{sys.executable} -m pip install --upgrade {pkg}"
            print(f"   Installing {pkg}...")
            subprocess.run(cmd, shell=True, capture_output=True)
            
    def run_full_diagnostic(self):
        """Run complete diagnostic"""
        self.check_windows_version()
        self.check_cpu_features()
        self.check_memory()
        self.check_visual_cpp_redist()
        self.check_gpu_drivers()
        self.check_python_packages()
        self.generate_report()
        
        if self.issues:
            response = input("\n🔧 Attempt automatic fixes? (y/n): ").strip().lower()
            if response == 'y':
                self.fix_common_issues()


def main():
    """Main execution"""
    diagnostic = SystemDiagnostic()
    diagnostic.run_full_diagnostic()
    
    print("\n" + "="*80)
    print("💡 NEXT STEPS:")
    print("="*80)
    print("1. Fix any identified issues")
    print("2. Install Visual C++ Redistributables if missing")
    print("3. Update Windows if on older build")
    print("4. Ensure 64-bit Python is installed")
    print("5. Run the diagnostic again after fixes")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()