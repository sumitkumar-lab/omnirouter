from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from src.rag.embeddings import get_embeddings_model
from src.rag.index_store import load_faiss_index
from src.rag.metadata_store import MetadataStore
from src.rag.pipeline import sync_corpus
from src.rag.settings import RagSettings, get_rag_settings


class EmptyRetriever(BaseRetriever):
    placeholder_message: str = Field(default="No indexed documents are available yet.")

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return []

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return []


def get_document_retriever(settings: RagSettings | None = None) -> BaseRetriever:
    settings = settings or get_rag_settings()
    sync_corpus(settings=settings)

    store = MetadataStore(settings.metadata_db_url)
    store.init_db()
    latest_index = store.get_latest_index()
    if latest_index is None:
        return EmptyRetriever()

    vector_store = load_faiss_index(Path(latest_index.index_path), get_embeddings_model(settings))
    return vector_store.as_retriever(search_kwargs={"k": settings.retrieval_k})
