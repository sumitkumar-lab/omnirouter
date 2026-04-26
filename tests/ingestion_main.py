from src.rag.pipeline import sync_corpus
from src.rag.vector_store import get_vector_store

def main():
    print("\n--- Step 1: Syncing source documents into versioned FAISS corpus ---")
    result = sync_corpus(force=True)
    print(f"Rebuilt: {result.rebuilt}, version: {result.version_label}, chunks: {result.chunk_count}")

    print("\n--- Step 2: Semantic Search Test ---")
    db = get_vector_store()
    query = "What happens if the main API goes down?"
    results = db.similarity_search(query, k=1)

    print(f"\nUser asked: '{query}'")
    if results:
        print(f"Vector DB found this relevant chunk: '{results[0].page_content}'")
    else:
        print("No matching content found. Add docs under 'knowledge_base/' or set RAG_DOCUMENTS_DIRS.")

if __name__ == "__main__":
    main()
