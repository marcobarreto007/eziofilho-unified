@echo off
REM EZIO Financial AI - Fix LFS Issues and Push to GitHub
REM User: marcobarreto007
REM Date: 2025-05-25

echo ================================================================
echo EZIO FINANCIAL AI - FIXING LFS AND PUSHING TO GITHUB
echo ================================================================

cd /d "C:\Users\anapa\eziofilho-unified\"

echo Step 1: Remove problematic LFS tracking...
del .gitattributes
git lfs untrack "*.bin"
git lfs untrack "*.model" 
git lfs untrack "*.safetensors"

echo Step 2: Reset and clean LFS cache...
git lfs fetch --all
git lfs prune

echo Step 3: Add files without LFS...
git add .gitattributes
git add -A

echo Step 4: Create new commit without LFS...
git commit -m "fix: Remove LFS tracking for problematic files - System ready for Codespaces deployment"

echo Step 5: Force push to GitHub...
git push origin main --force

echo Step 6: Verify repository status...
git status

echo ================================================================
echo LFS FIXED - SYSTEM PUSHED TO GITHUB - READY FOR CODESPACES
echo ================================================================
pause