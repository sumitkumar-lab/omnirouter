"""
Let's officially give your agent its brain.

We are going to use LangChain's @tool decorator. 
This magical little wrapper takes your standard Python function, 
reads the type hints (like query: str), reads the docstring, and automatically 
translates the entire thing into a strict JSON schema that OpenAI and Anthropic natively understand .
"""
import os
from langchain_core.tools import tool
from src.rag.vector_store import get_vector_store

# The @tool decorator converts this Python function into an LLM-readable JSON schema
@tool
def search_documentation(query: str) -> str:
    """
    Searches the internal engineering documentation for information about the OmniRouter, 
    LangChain, LangGraph, or general AI engineering concepts.
    
    Use this tool WHENEVER the user asks a technical question about how the system works, 
    fallback protocols, or specific coding architecture. Do NOT use this for general greetings.
    
    Args:
        query: The specific search term to look up in the database. 
               It should be a standalone, highly descriptive phrase.
    """
    print(f"\n--- [TOOL EXECUTION] Searching Vector DB for: '{query}' ---")
    
    # In production, ensure your API key is loaded securely
    api_key = os.getenv("OPENAI_API_KEY", "YOUR_REAL_OPENAI_API_KEY") 
    
    try:
        db = get_vector_store(api_key)
        
        # Retrieve the top 2 most relevant chunks
        results = db.similarity_search(query, k=2)
        
        if not results:
            return "No relevant information found in the documentation."
            
        # We must return a STRING, not a list of objects, so the LLM can read it easily
        combined_text = "\n\n".join([doc.page_content for doc in results])
        return combined_text
        
    except Exception as e:
        return f"Error executing search: {str(e)}"
    

if __name__ == "__main__":
    # Print the name the LLM sees
    print(f"Tool Name: {search_documentation.name}")
    
    # Print the description the LLM reads to make its decision
    print(f"\nTool Description: \n{search_documentation.description}")
    
    # Print the strict JSON schema the LLM must follow to use the tool
    print(f"\nTool Arguments Schema: \n{search_documentation.args}")