from src.rag.vector_store import build_vector_store
from langchain_core.documents import Document
import os

api_key = os.getenv("HF_TOKEN")

def seed_database():
    print("Seeding new HuggingFace database...")
    
    # 1. Our dummy text
    sample_text = (
        "OmniRouter is an enterprise-grade AI architecture combining high-concurrency "
        "LLM routing and local Vector Database retrieval. If the primary API fails, "
        "it seamlessly switches to a fallback model. It uses LangGraph for agentic reasoning."
    )
    
    # 2. Package it as a chunk
    doc = Document(page_content=sample_text, metadata={"source": "manual.pdf"})
    
    # 3. Build and save the DB
    build_vector_store([doc], api_key=api_key)

if __name__ == "__main__":
    seed_database()