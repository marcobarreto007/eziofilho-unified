# cuda_check.py - Verify CUDA and GPUs 
import torch 
import sys 
import os 
 
print("=" * 60) 
print("CUDA and GPU CHECK") 
print("=" * 60) 
 
# Check CUDA availability 
print(f"CUDA Available: {torch.cuda.is_available()}") 
if torch.cuda.is_available(): 
    print(f"CUDA Version: {torch.version.cuda}") 
    print(f"PyTorch Version: {torch.__version__}") 
    gpu_count = torch.cuda.device_count() 
    print(f"\nGPUs Found: {gpu_count}") 
    for i in range(gpu_count): 
        print(f"\n[GPU {i}]") 
        print(f"Name: {torch.cuda.get_device_name(i)}") 
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB") 
else: 
    print("NO CUDA GPUs FOUND!") 
    print("\nInstall PyTorch with CUDA:") 
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118") 
