import asyncio
import os
from src.router import OmniRouter
from src.schemas import RouterConfig

async def main():
    # 1. Initialize the router with your keys
    # In production, NEVER hardcode keys. Use os.getenv("OPENAI_API_KEY")
    api_keys = {
        "openai": "sk-proj-yp2fl1bF", # Put a fake key here to test the error loop
        "anthropic": "sk-ant-fake-anthropic-key-"
    }
    
    router = OmniRouter(api_keys=api_keys)
    
    # 2. Create our strict configuration
    # Remember, if we set temperature=5.0 here, Pydantic will block it!
    config = RouterConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        # temperature=0.7,
        max_retries=3,
        fallback_provider="anthropic",
        fallback_model="claude-3-haiku-20240307"
    )
    
    # prompt = "In one sentence, explain what a transformer model is."
    prompt = "Explain why fallback routing is important in software engineering."
    
    try:
        print("\n--- Sending Request to OmniRouter ---")
        # 3. Fire the async request
        result = await router.generate(prompt=prompt, config=config)
        
        print("\n--- Success! ---")
        print(f"Response: {result.content}")
        # print(f"Cost: ${result.cost_estimate}")
        # print(f"Tokens: {result.prompt_tokens} in / {result.completion_tokens} out")
        
    except Exception as e:
        print(f"\n--- Final Application Failure ---")
        print(f"Router could not complete request: {str(e)}")
        print(f"System completely crashed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

# import sys
# print("\n--- Diagnostic Check ---")
# print(f"Executing Python path: {sys.executable}")