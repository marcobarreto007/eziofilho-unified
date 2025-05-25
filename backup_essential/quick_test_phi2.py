import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '01_core_system'))

from core.local_model_wrapper import LocalModelWrapper

print("🚀 Testing Phi-2 GGUF Model...")

# Initialize model
model = LocalModelWrapper(
    model_path=r"C:\Users\anapa\.cache\models\phi-2.gguf",
    model_type="gguf",
    chat_enabled=True
)

print("✅ Model loaded successfully!")
print("\n💬 Chat with Phi-2 (type 'exit' to quit)")
print("-" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("👋 Goodbye!")
        break
    
    print("\nPhi-2: ", end="", flush=True)
    response = model.generate(user_input)
    print(response)