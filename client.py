import requests
import json
import os

def chat_with_agent(query: str):
    api_base = os.getenv("OMNIROUTER_API_BASE", "http://127.0.0.1:8000").rstrip("/")
    url = f"{api_base}/chat/stream"
    payload = {"query": query}
    
    print(f"\nYou: {query}")
    print("Agent: ", end="", flush=True) # Prepare the terminal line
    
    # We open a streaming connection to the FastAPI server
    with requests.post(url, json=payload, stream=True) as response:
        response.raise_for_status()
        event_type = "message"
        for line in response.iter_lines():
            if not line:
                event_type = "message"
                continue

            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("event: "):
                event_type = decoded_line.replace("event: ", "")
                continue
            if not decoded_line.startswith("data: "):
                continue

            try:
                data = json.loads(decoded_line.replace("data: ", ""))
            except json.JSONDecodeError:
                continue

            if event_type == "token":
                print(data.get("token", ""), end="", flush=True)
    print("\n") # Add a final line break when done

if __name__ == "__main__":
    # Make sure your FastAPI server is running in another terminal!
    # chat_with_agent("What is OmniRouter?")
    # chat_with_agent("Can you explain OmniRouter ?")
    # chat_with_agent("What are the top AI news headlines today?")
    # chat_with_agent("Calculate the 35th number in the Fibonacci sequence. Then multiply that number by 12.34.")
    # chat_with_agent("Please give me a quick summary of the bug reported in issue 20000 on the langchain-ai/langchain repo.")
    # chat_with_agent("Who were the top two performing employees by revenue across the whole company, and what departments are they in?")
    chat_with_agent("What is Chinchilla training?")
