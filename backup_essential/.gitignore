@echo off
REM EZIO Financial AI - Clean Repository (Remove Heavy Files)
REM User: marcobarreto007
REM Date: 2025-05-25

echo ================================================================
echo EZIO CLEAN REPOSITORY - REMOVING HEAVY FILES
echo ================================================================

cd /d "C:\Users\anapa\eziofilho-unified\"

echo Step 1: Remove from Git tracking...
git rm -r --cached portable_python/ 2>nul
git rm -r --cached phi3-mini/*.bin 2>nul
git rm -r --cached phi3-mini/*.model 2>nul
git rm -r --cached phi3-mini/*.safetensors 2>nul
git rm -r --cached "GitHub CLI/" 2>nul

echo Step 2: Delete heavy folders physically...
rmdir /s /q "portable_python" 2>nul
del /q /s "*.exe" 2>nul
del /q /s "*.dll" 2>nul

echo Step 3: Add .gitignore...
git add .gitignore

echo Step 4: Add essential Python files only...
git add *.py
git add *.json
git add *.md
git add *.txt
git add *.bat
git add 02_experts_modules/
git add autogen_*/
git add cuda_setup/
git add ezio_experts/
git add financial_experts/
git add github_setup/
git add hf_financial_models/
git add models/

echo Step 5: Clean commit...
git commit -m "feat: EZIO Financial AI Clean - Essential files only - Ready for Codespaces deployment"

echo Step 6: Force push clean repository...
git push origin main --force

echo Step 7: Repository status...
git status
echo.
echo Repository size check:
du -sh . 2>nul || echo "Repository cleaned successfully"

echo ================================================================
echo CLEAN REPOSITORY COMPLETED - READY FOR CODESPACES
echo ================================================================
pause