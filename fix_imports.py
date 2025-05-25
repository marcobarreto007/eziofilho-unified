#!/usr/bin/env python3
"""
Fix Imports - Add missing 'import os' statements
"""

import os
import re

def fix_imports_in_file(file_path):
    """Add import os if missing but os.getenv is used"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if 'import os' is needed but missing
        if 'os.getenv' in content and 'import os' not in content:
            lines = content.split('\n')
            
            # Find where to insert import os
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_index = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    # First non-comment, non-import line
                    insert_index = i
                    break
            
            # Insert import os
            lines.insert(insert_index, 'import os')
            content = '\n'.join(lines)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f'✅ Added import os to {file_path}')
            return True
        else:
            print(f'ℹ️  No import needed for {file_path}')
            return False
            
    except Exception as e:
        print(f'❌ Error fixing imports in {file_path}: {e}')
        return False

def main():
    """Fix imports in all cleaned files"""
    files_to_fix = [
        '03_models_storage/model_configs/download_mistral.py',
        '03_models_storage/model_configs/download_models.py',
        'backup_essential/test_hf_api.py',
        'ezio_experts/config/api_config.py',
        'test_hf_api.py'
    ]
    
    print("🔧 Fixing imports in cleaned files...")
    fixed_count = 0
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_imports_in_file(file_path):
                fixed_count += 1
        else:
            print(f'⚠️  File not found: {file_path}')
    
    print(f"\n✅ Import fix completed! Fixed {fixed_count} files.")

if __name__ == "__main__":
    main()