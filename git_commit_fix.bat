@echo off
REM EZIO Financial AI - Git Configuration Fix and Clean Commit
REM User: marcobarreto007
REM Date: 2025-05-25

echo ================================================================
echo EZIO FINANCIAL AI - FIXING GIT CONFIGURATION AND COMMIT
echo ================================================================

cd /d "C:\Users\anapa\eziofilho-unified\"

echo Configuring Git user...
git config user.name "Marco Barreto"
git config user.email "marcobarreto007@gmail.com"

echo Checking current status...
git status

echo Removing problematic LFS files temporarily...
git reset HEAD .
git lfs untrack "*.bin" "*.model" "*.safetensors"

echo Adding files in stages to avoid LFS issues...
git add README.md
git add requirements.txt
git add *.py
git add *.json
git add *.bat

echo Creating clean commit message...
git commit -m "feat: Complete EZIO Financial AI System - Enhanced trading system with 12 experts architecture - Multi-GPU Phi-3 integration - Real-time Yahoo Finance data - 111KB+ codebase ready for Codespaces"

echo Pushing to GitHub...
git push origin main

echo ================================================================
echo COMMIT FIX COMPLETED - Ready for GitHub Codespaces
echo ================================================================
pause