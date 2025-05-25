def setup_router(models):
    """Sets up the model router with available models."""
    # Import the necessary components
    _, ModelRouter = import_wrapper_and_router()
    from core.model_router import ModelDefinition, ModelCapability, create_model_router
    
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
    return router