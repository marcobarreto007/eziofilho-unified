import os
import time
from openai import OpenAI

# Configure client to use LM Studio's OpenAI-compatible API
client = OpenAI(
    base_url="http://localhost:1234/v1",  # LM Studio's default port
    api_key="lm-studio"  # Any string works here
)

def main():
    print("💬 LM Studio Client - Connected to local GPU model")
    print("Note: Make sure LM Studio is running with API server enabled\n")
    
    try:
        while True:
            user_input = input("Você: ").strip()
            if not user_input:
                continue
                
            start_time = time.time()
            
            # Call the model through LM Studio
            response = client.chat.completions.create(
                model="phi-2",  # Model name doesn't matter for LM Studio
                messages=[{"role": "user", "content": user_input}],
                max_tokens=256,
                temperature=0.7
            )
            
            elapsed = time.time() - start_time
            answer = response.choices[0].message.content
            
            print(f"🤖 {answer}")
            print(f"⏱️  Generation time: {elapsed:.2f}s\n")
            
    except KeyboardInterrupt:
        print("\n👋 Até a próxima!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check if LM Studio is running with API server enabled (Settings → API)")

if __name__ == "__main__":
    main()