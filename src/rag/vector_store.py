import os
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# This is where our local database will be saved on your hard drive
DB_DIRECTORY = "./chroma_db"

def build_vector_store(chunks: List[Document], api_key: str):
    """
    Takes a list of chunked documents, embeds them, and saves them to a local Chroma database.
    """
    print(f"Initializing embedding model...")
    # We use OpenAI's embedding model here. It converts text to 1536-dimensional vectors.
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    
    print(f"Embedding {len(chunks)} chunks and saving to {DB_DIRECTORY}...")
    
    # 1. Create the database
    # 2. Embed all the chunks
    # 3. Save it to the DB_DIRECTORY
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIRECTORY
    )
    
    # Force the database to save to disk
    vector_store.persist()
    print("Database successfully built and saved to disk!")
    return vector_store

def get_vector_store(api_key: str):
    """
    Retrieves the existing database from the hard drive so we don't have to rebuild it every time.
    """
    embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
    return Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)