import requests

base_url = "http://localhost:1234/v1"

# Simple test to check if server is responding
try:
    response = requests.get(f"{base_url}/models")
    print(f"✅ Connection successful! Models: {response.json()}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    
    # Suggestions
    print("\nTroubleshooting tips:")
    print("1. Make sure LM Studio is running")
    print("2. Check if API server is enabled (Settings → API)")
    print("3. Verify the port matches (default: 1234)")
    print("4. Try loading a model in LM Studio first")