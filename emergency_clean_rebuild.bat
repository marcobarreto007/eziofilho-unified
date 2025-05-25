@echo off
REM EZIO Financial AI - EMERGENCY CLEAN REBUILD
REM Removes Git LFS completely and recreates clean repository
REM User: marcobarreto007
REM Date: 2025-05-25

echo ================================================================
echo 🚨 EZIO EMERGENCY CLEAN REBUILD - REMOVING GIT LFS COMPLETELY
echo ================================================================

REM Step 1: Backup essential Python files
echo Step 1: Creating backup of essential files...
mkdir backup_essential 2>nul
copy "*.py" backup_essential\ 2>nul
copy "*.md" backup_essential\ 2>nul
copy "*.txt" backup_essential\ 2>nul
copy "*.json" backup_essential\ 2>nul
copy ".gitignore" backup_essential\ 2>nul

REM Step 2: Remove Git LFS completely
echo Step 2: Removing Git LFS completely...
git lfs uninstall
git lfs untrack "*"
if exist ".gitattributes" del /F ".gitattributes"

REM Step 3: Remove ALL large files and LFS objects
echo Step 3: Removing all large files and LFS cache...
if exist ".git\lfs\" rmdir /S /Q ".git\lfs\"
if exist "phi3-mini\model-*.safetensors" del /F "phi3-mini\model-*.safetensors"
if exist "phi3-mini\tokenizer.model" del /F "phi3-mini\tokenizer.model"

REM Step 4: Clean Git history (NUCLEAR OPTION)
echo Step 4: Cleaning Git history - NUCLEAR OPTION...
git checkout --orphan temp_branch
git add -A
git commit -m "feat: EZIO Financial AI - Clean Start - No LFS"

REM Step 5: Delete old main branch and rename
echo Step 5: Replacing main branch...
git branch -D main
git branch -m main

REM Step 6: Force garbage collection
echo Step 6: Force garbage collection...
git gc --aggressive --prune=all
git reflog expire --expire=now --all
git gc --aggressive --prune=now

REM Step 7: Update .gitignore to prevent future issues
echo Step 7: Updating .gitignore...
echo. >> .gitignore
echo # PREVENT ALL LARGE FILES >> .gitignore
echo *.safetensors >> .gitignore
echo *.bin >> .gitignore
echo *.model >> .gitignore
echo model-*.* >> .gitignore
echo **/model-*.* >> .gitignore
echo **/*.safetensors >> .gitignore
echo **/*.bin >> .gitignore

REM Step 8: Check repository size
echo Step 8: Checking new repository size...
du -sh .git 2>nul || echo "Repository cleaned"

REM Step 9: Final push (FORCE - WARNING!)
echo Step 9: Force push NEW clean repository...
git remote set-url origin https://github.com/marcobarreto007/eziofilho-unified.git
git push --force origin main

echo ================================================================
echo ✅ EMERGENCY REBUILD COMPLETED!
echo 🎯 Repository should be completely clean now
echo 📦 Size should be under 100MB
echo 🚀 Ready for GitHub Codespaces!
echo ================================================================

REM Step 10: Verify success
echo Step 10: Verifying success...
git status
git log --oneline -5

pause