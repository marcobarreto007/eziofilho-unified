# check_configure_cuda.py - Check and configure CUDA for PyTorch
# Audit Mode: Active - CUDA detection and configuration
# Path: C:\Users\anapa\EzioFilhoUnified\ezio_experts\cuda_setup
# User: marcobarreto007
# Date: 2025-05-24 21:11:00 UTC
# Objective: Detect CUDA and install correct PyTorch version

import os
import sys
import subprocess
import platform
from pathlib import Path

print("=" * 80)
print("🔍 CUDA DETECTION AND CONFIGURATION")
print("=" * 80)

def check_nvidia_gpu():
    """Check if NVIDIA GPU is present"""
    print("\n📊 Checking for NVIDIA GPU...")
    
    try:
        # Try nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            print("\nGPU Info:")
            lines = result.stdout.split('\n')
            for line in lines[:15]:  # First 15 lines contain GPU info
                if line.strip():
                    print(line)
            return True
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    
    # Check Windows Device Manager
    try:
        result = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            capture_output=True, text=True
        )
        if 'NVIDIA' in result.stdout:
            print("✅ NVIDIA GPU found in Device Manager")
            gpu_lines = [line.strip() for line in result.stdout.split('\n') if 'NVIDIA' in line]
            for gpu in gpu_lines:
                print(f"  • {gpu}")
            return True
    except:
        pass
    
    return False

def check_cuda_installation():
    """Check CUDA installation"""
    print("\n🔍 Checking CUDA installation...")
    
    cuda_paths = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        Path("C:/Program Files/NVIDIA Corporation/CUDA"),
        Path("C:/CUDA")
    ]
    
    cuda_version = None
    for cuda_path in cuda_paths:
        if cuda_path.exists():
            # Look for version folders
            versions = [d for d in cuda_path.iterdir() if d.is_dir() and d.name.startswith('v')]
            if versions:
                cuda_version = versions[-1].name  # Get latest version
                print(f"✅ CUDA found: {cuda_path / cuda_version}")
                return cuda_version
    
    # Check environment variable
    cuda_path_env = os.environ.get('CUDA_PATH')
    if cuda_path_env:
        print(f"✅ CUDA_PATH: {cuda_path_env}")
        return Path(cuda_path_env).name
    
    print("❌ CUDA not found")
    return None

def check_pytorch_cuda():
    """Check PyTorch CUDA support"""
    print("\n🔍 Checking PyTorch CUDA support...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            
            return True
        else:
            print("❌ PyTorch CUDA support not available")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def get_cuda_version_for_pytorch():
    """Determine correct CUDA version for PyTorch"""
    cuda_version = check_cuda_installation()
    
    if cuda_version:
        # Extract version number
        version_num = cuda_version.replace('v', '').replace('.', '')[:3]
        
        # Map to PyTorch CUDA versions
        cuda_map = {
            '118': 'cu118',  # CUDA 11.8
            '117': 'cu117',  # CUDA 11.7
            '116': 'cu116',  # CUDA 11.6
            '121': 'cu121',  # CUDA 12.1
            '120': 'cu118'   # CUDA 12.0 -> use 11.8
        }
        
        return cuda_map.get(version_num, 'cu118')  # Default to 11.8
    
    return None

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("\n📦 Installing PyTorch with CUDA support...")
    
    cuda_version = get_cuda_version_for_pytorch()
    
    if not cuda_version:
        print("❌ Cannot determine CUDA version")
        print("Installing CPU-only version...")
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    else:
        print(f"Installing PyTorch for CUDA {cuda_version}")
        # Uninstall existing PyTorch first
        print("Removing existing PyTorch...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        
        # Install CUDA version
        index_url = f"https://download.pytorch.org/whl/{cuda_version}"
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url
        ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✅ PyTorch installation complete!")
    else:
        print("❌ Installation failed")

def test_cuda_performance():
    """Test CUDA performance"""
    print("\n🚀 Testing CUDA performance...")
    
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available for testing")
            return
        
        # Create test tensors
        size = 10000
        device = torch.device('cuda')
        
        print(f"\nMatrix multiplication test ({size}x{size})...")
        
        # CPU test
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        print(f"CPU time: {cpu_time:.3f} seconds")
        
        # GPU test
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        
        # Warm up
        torch.cuda.synchronize()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"GPU time: {gpu_time:.3f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

def main():
    """Main execution"""
    # System info
    print(f"\n💻 System Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
    # Check components
    has_gpu = check_nvidia_gpu()
    cuda_version = check_cuda_installation()
    has_pytorch_cuda = check_pytorch_cuda()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    print("="*60)
    print(f"NVIDIA GPU: {'✅ Yes' if has_gpu else '❌ No'}")
    print(f"CUDA Toolkit: {'✅ ' + str(cuda_version) if cuda_version else '❌ Not installed'}")
    print(f"PyTorch CUDA: {'✅ Enabled' if has_pytorch_cuda else '❌ Disabled'}")
    
    # Recommendations
    if has_gpu and not has_pytorch_cuda:
        print("\n⚠️  GPU detected but PyTorch CUDA not enabled!")
        response = input("\nInstall PyTorch with CUDA support? (y/n): ").strip().lower()
        if response == 'y':
            install_pytorch_cuda()
            # Test again
            print("\n🔄 Rechecking PyTorch CUDA...")
            check_pytorch_cuda()
            test_cuda_performance()
    elif has_pytorch_cuda:
        print("\n✅ Everything is configured correctly!")
        response = input("\nRun performance test? (y/n): ").strip().lower()
        if response == 'y':
            test_cuda_performance()
    else:
        print("\n❌ No NVIDIA GPU detected. Using CPU-only PyTorch.")
        print("\nTo use GPU acceleration, you need:")
        print("1. NVIDIA GPU (GTX 1060 or better recommended)")
        print("2. CUDA Toolkit (11.8 or 12.1)")
        print("3. PyTorch with CUDA support")

if __name__ == "__main__":
    main()