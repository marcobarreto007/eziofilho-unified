import requests

try:
    response = requests.get("http://localhost:1234/v1/models")
    print(f"Connected to LM Studio! Available models:")
    print(response.json())
except Exception as e:
    print(f"Connection failed: {e}")
    print("\nMake sure:")
    print("1. LM Studio is running")
    print("2. Server is enabled (Menu → Server → Run Server)")
    print("3. A model is loaded")