# simple_diagnostic.py - Simple system check
# Path: C:\Users\anapa\eziofilho-unified\
# Objective: Basic compatibility check without complex imports

import sys
import platform
import os

print("=" * 60)
print("SIMPLE SYSTEM DIAGNOSTIC")
print("=" * 60)

# Basic info
print(f"\nPython: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")

# Check if 64-bit
is_64bit = sys.maxsize > 2**32
print(f"64-bit Python: {'Yes' if is_64bit else 'No'}")

# Check paths
print(f"\nPython executable: {sys.executable}")
print(f"Python prefix: {sys.prefix}")

# Check environment
print(f"\nPATH includes Python: {'python' in os.environ.get('PATH', '').lower()}")

# Try imports
print("\nTesting imports:")
modules = ['os', 'sys', 'json', 'pathlib', 'datetime']
for mod in modules:
    try:
        __import__(mod)
        print(f"  ✓ {mod}")
    except:
        print(f"  ✗ {mod}")

print("\nPress Enter to exit...")
input()