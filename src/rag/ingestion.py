from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_document_text(raw_text: str):
    """
    Simulates taking a massive document and chunking it for a Vector Store.
    """
    print(f"Original Document Length: {len(raw_text)} characters")
    
    # THE CHUNKER CONFIGURATION
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,       # The maximum size of each chunk
        chunk_overlap=20,     # How much the chunks should overlap
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Tries to split at paragraphs first, then sentences
    )
    
    # Create a LangChain Document object
    doc = Document(page_content=raw_text, metadata={"source": "engineering_manual.pdf"})
    
    # Execute the split
    chunks = text_splitter.split_documents([doc])
    
    print(f"\nCreated {len(chunks)} chunks.")
    
    # Let's inspect the exact output to understand the data structure
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)

    return chunks
# Let's test it with a sample "manual"
if __name__ == "__main__":
    sample_manual = (
        "OmniRouter is an advanced asynchronous LLM routing engine. "
        "It is designed to handle multiple providers gracefully. "
        "If the primary provider fails, the system initiates a failover protocol. "
        "This ensures maximum uptime for production systems."
    )
    
    chunk_document_text(sample_manual)