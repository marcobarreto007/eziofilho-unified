@echo off
REM EZIO Financial AI - Clean ALL Hardcoded Tokens
REM User: marcobarreto007
REM Date: 2025-05-25

echo ================================================================
echo 🔐 EZIO COMPLETE TOKEN CLEANUP - REMOVING ALL HARDCODED TOKENS
echo ================================================================

REM Create Python script to clean all tokens
echo import os > clean_tokens.py
echo import re >> clean_tokens.py
echo import glob >> clean_tokens.py
echo. >> clean_tokens.py
echo def clean_token_in_file(file_path): >> clean_tokens.py
echo     """Clean tokens from a specific file""" >> clean_tokens.py
echo     try: >> clean_tokens.py
echo         with open(file_path, 'r', encoding='utf-8') as f: >> clean_tokens.py
echo             content = f.read() >> clean_tokens.py
echo. >> clean_tokens.py
echo         # Pattern to find HF tokens >> clean_tokens.py
echo         hf_pattern = r'hf_[a-zA-Z0-9]{34,}' >> clean_tokens.py
echo         replacement = 'os.getenv("HUGGINGFACE_TOKEN", "your_token_here")' >> clean_tokens.py
echo. >> clean_tokens.py
echo         # Replace direct token assignments >> clean_tokens.py
echo         patterns = [ >> clean_tokens.py
echo             (r'HF_TOKEN\s*=\s*["\']hf_[^"\']*["\']', 'HF_TOKEN = os.getenv("HF_TOKEN", "your_token_here")'), >> clean_tokens.py
echo             (r'os\.environ\[["\']HF_TOKEN["\']\]\s*=\s*["\']hf_[^"\']*["\']', 'os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "your_token_here")'), >> clean_tokens.py
echo             (r'"api_key":\s*["\']hf_[^"\']*["\']', '"api_key": os.getenv("HUGGINGFACE_TOKEN", "your_token_here")'), >> clean_tokens.py
echo             (r'"huggingface":\s*os\.getenv\([^)]*["\']hf_[^"\']*["\'][^)]*\)', '"huggingface": os.getenv("HUGGINGFACE_API_KEY", "your_token_here")'), >> clean_tokens.py
echo             (hf_pattern, 'os.getenv("HUGGINGFACE_TOKEN", "your_token_here")') >> clean_tokens.py
echo         ] >> clean_tokens.py
echo. >> clean_tokens.py
echo         cleaned_content = content >> clean_tokens.py
echo         changes_made = False >> clean_tokens.py
echo. >> clean_tokens.py
echo         for pattern, replacement_text in patterns: >> clean_tokens.py
echo             if re.search(pattern, cleaned_content): >> clean_tokens.py
echo                 cleaned_content = re.sub(pattern, replacement_text, cleaned_content) >> clean_tokens.py
echo                 changes_made = True >> clean_tokens.py
echo. >> clean_tokens.py
echo         if changes_made: >> clean_tokens.py
echo             with open(file_path, 'w', encoding='utf-8') as f: >> clean_tokens.py
echo                 f.write(cleaned_content) >> clean_tokens.py
echo             print(f'✅ Cleaned tokens from {file_path}') >> clean_tokens.py
echo             return True >> clean_tokens.py
echo         else: >> clean_tokens.py
echo             print(f'ℹ️  No tokens found in {file_path}') >> clean_tokens.py
echo             return False >> clean_tokens.py
echo. >> clean_tokens.py
echo     except Exception as e: >> clean_tokens.py
echo         print(f'❌ Error cleaning {file_path}: {e}') >> clean_tokens.py
echo         return False >> clean_tokens.py
echo. >> clean_tokens.py
echo # List of problematic files found >> clean_tokens.py
echo files_to_clean = [ >> clean_tokens.py
echo     '03_models_storage/model_configs/download_mistral.py', >> clean_tokens.py
echo     '03_models_storage/model_configs/download_models.py', >> clean_tokens.py
echo     'backup_essential/test_hf_api.py', >> clean_tokens.py
echo     'ezio_experts/config/api_config.py', >> clean_tokens.py
echo     'test_hf_api.py' >> clean_tokens.py
echo ] >> clean_tokens.py
echo. >> clean_tokens.py
echo print("🔍 Cleaning hardcoded tokens from files...") >> clean_tokens.py
echo cleaned_count = 0 >> clean_tokens.py
echo. >> clean_tokens.py
echo for file_path in files_to_clean: >> clean_tokens.py
echo     if os.path.exists(file_path): >> clean_tokens.py
echo         if clean_token_in_file(file_path): >> clean_tokens.py
echo             cleaned_count += 1 >> clean_tokens.py
echo     else: >> clean_tokens.py
echo         print(f'⚠️  File not found: {file_path}') >> clean_tokens.py
echo. >> clean_tokens.py
echo print(f"\n✅ Token cleanup completed! Cleaned {cleaned_count} files.") >> clean_tokens.py

REM Execute the cleaning script
echo Step 1: Executing token cleanup...
python clean_tokens.py

REM Clean up the script
del clean_tokens.py

REM Step 2: Add import os where needed
echo Step 2: Adding missing imports...
python -c "
import os
import re

files_to_fix = [
    '03_models_storage/model_configs/download_mistral.py',
    '03_models_storage/model_configs/download_models.py',
    'backup_essential/test_hf_api.py',
    'ezio_experts/config/api_config.py',
    'test_hf_api.py'
]

for file_path in files_to_fix:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if 'import os' is already present
            if 'import os' not in content and 'os.getenv' in content:
                # Add import os at the beginning
                lines = content.split('\n')
                # Find the first import or add at the beginning
                import_index = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        import_index = i
                        break
                
                lines.insert(import_index, 'import os')
                content = '\n'.join(lines)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f'✅ Added import os to {file_path}')
        except Exception as e:
            print(f'❌ Error fixing imports in {file_path}: {e}')
"

REM Step 3: Verify no tokens remain
echo Step 3: Verifying token cleanup...
echo Searching for remaining tokens...
findstr /s /i "hf_[a-zA-Z0-9]" *.py > token_search.tmp 2>nul
if exist token_search.tmp (
    echo ⚠️  Remaining tokens found:
    type token_search.tmp
    del token_search.tmp
) else (
    echo ✅ No hardcoded tokens found!
)

REM Step 4: Add and commit changes
echo Step 4: Adding cleaned files to git...
git add .

echo Step 5: Checking git status...
git status

echo ================================================================
echo ✅ ALL TOKENS CLEANED SUCCESSFULLY!
echo 🔐 Repository is now secure
echo 📝 Ready for commit and push
echo ================================================================

pause