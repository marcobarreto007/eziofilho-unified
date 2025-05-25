import shutil
from datetime import datetime

# Make backup first
backup_name = f"01_core_system/run_system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
shutil.copy("01_core_system/run_system.py", backup_name)
print(f"Backup created: {backup_name}")

# Read the original file
with open("01_core_system/run_system.py", "r", encoding="utf-8") as f:
    content = f.read()

# The corrected setup_router function
new_function = '''def setup_router(models):
    """Sets up the model router with available models."""
    # Import the necessary components
    _, ModelRouter = import_wrapper_and_router()
    from core.model_router import create_model_router
    
    # Create model configurations for the router
    model_configs = []
    
    for name, wrapper in models.items():
        # Determine capabilities based on model name
        capabilities = ["general"]  # Default capability
        
        if "phi" in name.lower():
            capabilities.extend(["fast", "creative"])
        elif "mistral" in name.lower():
            capabilities.extend(["precise", "code"])
        elif "dialogpt" in name.lower():
            capabilities.extend(["creative", "conversation"])
        
        # Create model configuration
        config = {
            "name": name,
            "path": wrapper.model_path if hasattr(wrapper, 'model_path') else name,
            "model_type": wrapper.model_type if hasattr(wrapper, 'model_type') else "gguf",
            "capabilities": capabilities,
            "min_prompt_tokens": 0,
            "max_prompt_tokens": 2048
        }
        
        model_configs.append(config)
    
    # Create router using the factory function
    router = create_model_router(
        model_configs=model_configs,
        default_model=list(models.keys())[0] if models else None
    )
    
    logger.info(f"Router configured with {len(models)} models")
    return router'''

# Find and replace the function
start = content.find('def setup_router(models):')
end = content.find('\n# ==============================================================================', start)

if start != -1 and end != -1:
    # Replace the function
    new_content = content[:start] + new_function + content[end:]
    
    # Write the fixed file
    with open("01_core_system/run_system.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("✅ File fixed successfully!")
else:
    print("❌ Could not find the function to replace")