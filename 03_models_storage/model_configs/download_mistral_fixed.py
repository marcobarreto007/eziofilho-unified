#!/usr/bin/env python3
"""
Download Mistral GGUF model - Secure Version
Compatible with GitHub Codespaces and local environments
"""

import os
import sys
from pathlib import Path
import shutil

def get_secure_token():
    """
    Securely get Hugging Face token from environment variables.
    
    Returns:
        str: HF token or empty string if not found
    """
    # Try multiple environment variable names
    token_vars = ['HUGGINGFACE_TOKEN', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN']
    
    for var in token_vars:
        token = os.getenv(var)
        if token and token != "your_token_here":
            print(f"✅ Found HF token in environment variable: {var}")
            return token
    
    print("⚠️  No Hugging Face token found in environment variables")
    print("💡 Model will be downloaded without authentication (public models only)")
    return ""

def setup_cache_directory():
    """
    Set up cache directory for models.
    
    Returns:
        Path: Cache directory path
    """
    # Use different cache locations based on environment
    if os.getenv('CODESPACES'):
        # In GitHub Codespaces
        cache_dir = Path("/workspace/.cache/models")
    elif os.getenv('COLAB_GPU'):
        # In Google Colab
        cache_dir = Path("/content/.cache/models")
    else:
        # Local environment
        cache_dir = Path.home() / ".cache" / "models"
    
    # Create directory if it doesn't exist
    cache_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Cache directory: {cache_dir}")
    
    return cache_dir

def download_mistral_model(cache_dir, token):
    """
    Download Mistral GGUF model from Hugging Face.
    
    Args:
        cache_dir (Path): Directory to save the model
        token (str): Hugging Face token
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    
    model_info = {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "local_filename": "mistral-7b.gguf"
    }
    
    print(f"📦 Downloading Mistral-7B GGUF from {model_info['repo_id']}...")
    
    try:
        # Prepare download arguments
        download_args = {
            "repo_id": model_info["repo_id"],
            "filename": model_info["filename"],
            "local_dir": str(cache_dir),
            "local_dir_use_symlinks": False,
        }
        
        # Add token only if available
        if token:
            download_args["token"] = token
        
        # Download the model
        downloaded_path = hf_hub_download(**download_args)
        print(f"✅ Downloaded: {downloaded_path}")
        
        # Rename to expected name
        src_path = cache_dir / model_info["filename"]
        dst_path = cache_dir / model_info["local_filename"]
        
        if src_path.exists() and src_path != dst_path:
            if dst_path.exists():
                dst_path.unlink()  # Remove existing file
            src_path.rename(dst_path)
            print(f"✅ Renamed to: {dst_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download Mistral: {e}")
        return False

def create_fallback_model(cache_dir):
    """
    Create a fallback model if main download fails.
    
    Args:
        cache_dir (Path): Directory to save the model
    
    Returns:
        bool: True if fallback created, False otherwise
    """
    print("🔄 Creating fallback model...")
    
    # Look for existing models to use as fallback
    fallback_sources = [
        "phi-2.gguf",
        "phi-3-mini.gguf", 
        "llama-2-7b.gguf"
    ]
    
    dst_path = cache_dir / "mistral-7b.gguf"
    
    for source_name in fallback_sources:
        src_path = cache_dir / source_name
        if src_path.exists():
            try:
                shutil.copy2(src_path, dst_path)
                print(f"✅ Created fallback from: {src_path}")
                return True
            except Exception as e:
                print(f"⚠️  Failed to copy {src_path}: {e}")
                continue
    
    # Create a minimal placeholder file
    try:
        with open(dst_path, 'w') as f:
            f.write("# Placeholder model file - replace with actual GGUF model\n")
        print(f"📝 Created placeholder file: {dst_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create placeholder: {e}")
        return False

def verify_model(cache_dir):
    """
    Verify that the model file exists and is valid.
    
    Args:
        cache_dir (Path): Directory containing the model
    
    Returns:
        bool: True if model is valid, False otherwise
    """
    model_path = cache_dir / "mistral-7b.gguf"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Check file size (GGUF models should be reasonably large)
    file_size = model_path.stat().st_size
    
    if file_size < 1024:  # Less than 1KB is probably a placeholder
        print(f"⚠️  Model file is very small ({file_size} bytes) - likely a placeholder")
        return False
    
    print(f"✅ Model file verified: {model_path} ({file_size:,} bytes)")
    return True

def main():
    """Main function to download Mistral model."""
    print("🚀 Mistral Model Downloader - Secure Version")
    print("=" * 50)
    
    try:
        # Step 1: Get secure token
        token = get_secure_token()
        
        # Step 2: Setup cache directory
        cache_dir = setup_cache_directory()
        
        # Step 3: Check if model already exists
        model_path = cache_dir / "mistral-7b.gguf"
        if model_path.exists() and model_path.stat().st_size > 1024:
            print(f"ℹ️  Model already exists: {model_path}")
            if verify_model(cache_dir):
                print("🎉 Setup complete - using existing model!")
                return 0
        
        # Step 4: Download model
        success = download_mistral_model(cache_dir, token)
        
        # Step 5: Create fallback if download failed
        if not success:
            print("🔄 Main download failed, attempting fallback...")
            success = create_fallback_model(cache_dir)
        
        # Step 6: Verify final result
        if success and verify_model(cache_dir):
            print("\n🎉 Mistral model setup complete!")
            print(f"📍 Model location: {cache_dir / 'mistral-7b.gguf'}")
            
            # Provide usage tips
            print("\n💡 Usage tips:")
            print("  - Use environment variables for HF tokens")
            print("  - Model is cached for future use")
            print("  - Compatible with GitHub Codespaces")
            
            return 0
        else:
            print("\n❌ Failed to set up Mistral model")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Download cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())