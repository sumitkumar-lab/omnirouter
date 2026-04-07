import asyncio
import os
from src.rag.ingestion import chunk_document_text
from src.rag.vector_store import build_vector_store

def main():
    # # In production, use os.getenv("OPENAI_API_KEY")
    # api_key = "YOUR_REAL_OPENAI_API_KEY"

    # Huggingface api_key
    api_key = os.getenv("HF_TOKEN")
    
    # 1. The Raw Data
    sample_manual = (
        "OmniRouter is an advanced asynchronous LLM routing engine. "
        "It is designed to handle multiple providers gracefully. "
        "If the primary provider fails, the system initiates a failover protocol. "
        "The failover protocol defaults to Anthropic's Claude model. "
        "This ensures maximum uptime for production systems."
    )
    
    # 2. Chunk the text
    print("\n--- Step 1: Chunking Document ---")
    chunks = chunk_document_text(sample_manual)
    
    # 3. Embed and Store
    print("\n--- Step 2: Building Vector Database ---")
    # This creates a folder called 'chroma_db' in your project
    db = build_vector_store(chunks, api_key=api_key)
    
    # 4. The Magic: Let's do a mathematical search!
    print("\n--- Step 3: Semantic Search Test ---")
    query = "What happens if the main API goes down?"
    
    # 'k=1' means return the top 1 most relevant chunk
    results = db.similarity_search(query, k=1) 
    
    print(f"\nUser asked: '{query}'")
    print(f"Vector DB found this relevant chunk: '{results[0].page_content}'")

if __name__ == "__main__":
    main()