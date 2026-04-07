from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CACHE_DIR = "./semantic_cache_db"

def get_cache_db():
    """Initializes the Cache Database using free local embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=CACHE_DIR, embedding_function=embeddings)

def check_cache(query: str, threshold: float = 0.5) -> str | None:
    """
    Embeds the user's question and mathematically checks if anyone 
    has asked a highly similar question before.
    """
    db = get_cache_db()
    
    # We search the cache and ask for the 'distance score'
    results = db.similarity_search_with_score(query, k=1)
    
    if results:
        doc, score = results[0]
        # In ChromaDB's default math (L2 distance), a LOWER score means it's MORE similar.
        # 0.0 is an exact match. 0.5 means "very similar meaning".
        if score < threshold:
            print(f"\n🟢 [CACHE HIT] Similar question found! (Score: {score:.3f})")
            return doc.metadata.get("answer")
            
    print("\n🔴 [CACHE MISS] Question is new.")
    return None

def save_to_cache(query: str, answer: str):
    """Saves a brand new question and its answer into the database."""
    db = get_cache_db()
    
    # The 'page_content' is the question. The 'metadata' holds the answer.
    doc = Document(page_content=query, metadata={"answer": answer})
    db.add_documents([doc])
    
    print("\n💾 [CACHE SAVED] New interaction stored for future users.")