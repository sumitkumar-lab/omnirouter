from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.rag.embeddings import get_embeddings_model
from src.rag.index_store import build_faiss_index, load_faiss_index
from src.rag.metadata_store import MetadataStore
from src.rag.pipeline import sync_corpus
from src.rag.settings import RagSettings, get_rag_settings


class EmptyVectorStore:
    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return []

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None):
        from src.rag.retrieval import EmptyRetriever

        return EmptyRetriever()


def build_vector_store(chunks: list[Document], api_key: str | None = None, settings: RagSettings | None = None):
    settings = settings or get_rag_settings()
    manual_index_path = settings.corpus_dir / "manual_build" / "faiss_index"
    if manual_index_path.parent.exists():
        shutil.rmtree(manual_index_path.parent)
    if not chunks:
        return EmptyVectorStore()
    build_faiss_index(chunks, get_embeddings_model(settings), manual_index_path)
    return load_faiss_index(manual_index_path, get_embeddings_model(settings))


def get_vector_store(api_key: str | None = None, auto_sync: bool = True, settings: RagSettings | None = None):
    settings = settings or get_rag_settings()
    if auto_sync:
        sync_corpus(settings=settings)

    store = MetadataStore(settings.metadata_db_url)
    store.init_db()
    latest_index = store.get_latest_index()
    if latest_index is None:
        return EmptyVectorStore()

    return load_faiss_index(Path(latest_index.index_path), get_embeddings_model(settings))
