# check_repository_status.py - Check GitHub repository status
# Audit Mode: Active - Repository verification
# Path: C:\Users\anapa\eziofilho-unified
# User: marcobarreto007
# Date: 2025-05-24 20:29:32 UTC
# Objective: Verify repository status and pending tasks

import os
import subprocess
import requests
from pathlib import Path
import json

print("=" * 70)
print("🔍 CHECKING REPOSITORY STATUS")
print("=" * 70)

# Change to project directory
os.chdir(r"C:\Users\anapa\eziofilho-unified")

# Check git status
print("\n📊 LOCAL GIT STATUS:")
print("-" * 50)
result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
if result.stdout:
    print("⚠️  Uncommitted changes:")
    print(result.stdout)
else:
    print("✅ Working directory clean")

# Check remote status
print("\n🌐 REMOTE REPOSITORY STATUS:")
print("-" * 50)
result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
print(result.stdout)

# Check for unpushed commits
result = subprocess.run("git log origin/main..HEAD --oneline", shell=True, capture_output=True, text=True)
if result.stdout:
    print("\n⚠️  Unpushed commits:")
    print(result.stdout)
else:
    print("✅ All commits pushed")

# Check GitHub API
print("\n🌍 GITHUB API CHECK:")
print("-" * 50)
try:
    response = requests.get("https://api.github.com/repos/marcobarreto007/eziofilho-unified")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Repository: {data['full_name']}")
        print(f"📊 Size: {data['size']} KB")
        print(f"⭐ Stars: {data['stargazers_count']}")
        print(f"👁️ Watchers: {data['watchers_count']}")
        print(f"🍴 Forks: {data['forks_count']}")
        print(f"📅 Created: {data['created_at']}")
        print(f"🔄 Updated: {data['updated_at']}")
        print(f"🌐 URL: {data['html_url']}")
    else:
        print(f"❌ Could not access repository (Status: {response.status_code})")
except Exception as e:
    print(f"❌ API Error: {e}")

# Check for large files
print("\n📦 CHECKING FOR LARGE FILES:")
print("-" * 50)
large_files = []
for file in Path(".").rglob("*"):
    if file.is_file():
        try:
            size_mb = file.stat().st_size / 1024 / 1024
            if size_mb > 50:  # Files larger than 50MB
                large_files.append((file, size_mb))
        except:
            pass

if large_files:
    print("⚠️  Large files found:")
    for file, size in large_files:
        print(f"   - {file}: {size:.1f}MB")
else:
    print("✅ No large files found")

# Check LFS status
print("\n🔍 GIT LFS STATUS:")
print("-" * 50)
result = subprocess.run("git lfs ls-files", shell=True, capture_output=True, text=True)
if result.stdout:
    print("⚠️  LFS tracked files:")
    print(result.stdout)
else:
    print("✅ No LFS files")

# Create action plan
print("\n" + "="*70)
print("📋 ACTION PLAN:")
print("="*70)

actions_needed = []

# Check if we need to remove LFS
if result.stdout or large_files:
    actions_needed.append("Remove large files and disable LFS")

# Check for uncommitted changes
result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
if result.stdout:
    actions_needed.append("Commit pending changes")

# Check for unpushed commits
result = subprocess.run("git log origin/main..HEAD --oneline", shell=True, capture_output=True, text=True)
if result.stdout:
    actions_needed.append("Push commits to GitHub")

if actions_needed:
    print("\n⚠️  Actions needed:")
    for i, action in enumerate(actions_needed, 1):
        print(f"{i}. {action}")
    
    print("\n🔧 AUTOMATED FIX COMMANDS:")
    print("-" * 50)
    
    # Generate fix commands
    if any("large files" in a for a in actions_needed):
        print("# Remove large files:")
        print("git lfs uninstall")
        print("git rm -r --cached .")
        print("del /s *.bin *.safetensors *.gguf")
        print("git add .")
        print('git commit -m "fix: Remove large files"')
    
    if any("Commit" in a for a in actions_needed):
        print("\n# Commit changes:")
        print("git add .")
        print('git commit -m "update: Latest changes"')
    
    if any("Push" in a for a in actions_needed):
        print("\n# Push to GitHub:")
        print("git push -u origin main --force")
else:
    print("\n✅ REPOSITORY IS PERFECT!")
    print("\n🎉 Your repository is ready to use in GitHub Codespaces!")
    print("\n🚀 Next steps:")
    print("1. Go to: https://github.com/marcobarreto007/eziofilho-unified")
    print("2. Click 'Code' → 'Codespaces' → 'Create codespace on main'")
    print("3. Enjoy your AI system in the cloud!")

# Show repository link
print("\n" + "="*70)
print(f"🌐 Repository: https://github.com/marcobarreto007/eziofilho-unified")
print("="*70)

input("\nPress Enter to exit...")