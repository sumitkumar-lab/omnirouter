import os
from langchain_core.messages import HumanMessage, AIMessage
from src.rag.chatbot import build_doc_assistant

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("Booting up Documentation Assistant...")
    rag_chain = build_doc_assistant(api_key)
    
    # This list is our ConversationBufferMemory!
    # We store the history here and pass it to the chain on every loop.
    chat_history = []
    
    print("\n--- Assistant Ready. Type 'quit' to exit. ---")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        # Execute the complex chain
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        answer = response["answer"]
        print(f"\nAssistant: {answer}")
        
        # Update our memory for the next loop
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

if __name__ == "__main__":
    main()