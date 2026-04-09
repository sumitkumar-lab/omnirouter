import requests
import json

def chat_with_agent(query: str):
    # url = "http://127.0.0.1:8000/chat/stream"
    url = "https://YOUR_USERNAME-omnirouter-api.hf.space/chat/stream"
    payload = {"query": query}
    
    print(f"\nYou: {query}")
    print("Agent: ", end="", flush=True) # Prepare the terminal line
    
    # We open a streaming connection to the FastAPI server
    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # Look for the SSE "data: " prefix
                if decoded_line.startswith("data: "):
                    # Strip out the "data: " part so we just have pure JSON
                    json_str = decoded_line.replace("data: ", "")
                    try:
                        data = json.loads(json_str)
                        token = data.get("token", "")
                        # Print the token immediately on the same line without a line break
                        print(token, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
    print("\n") # Add a final line break when done

if __name__ == "__main__":
    # Make sure your FastAPI server is running in another terminal!
    chat_with_agent("What is OmniRouter?")
    # chat_with_agent("Can you explain OmniRouter ?")