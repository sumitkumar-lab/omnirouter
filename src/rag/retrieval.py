from __future__ import annotations

import re
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


class SourceAwareRetriever(BaseRetriever):
    vector_store: Any
    k: int = Field(default=4)
    lexical_pool_size: int = Field(default=12)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        semantic_docs = self.vector_store.similarity_search(
            query,
            k=max(self.k, self.lexical_pool_size),
        )
        lexical_docs = self._lexical_matches(query)
        docs = _dedupe_documents([*lexical_docs, *semantic_docs])
        docs.sort(key=lambda doc: _quality_score(doc, _query_terms(query)), reverse=True)
        return docs[: self.k]

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)

    def _lexical_matches(self, query: str) -> list[Document]:
        terms = _query_terms(query)
        if not terms:
            return []

        scored_docs: list[tuple[int, Document]] = []
        for doc in _all_indexed_documents(self.vector_store):
            score = _lexical_score(doc, terms)
            if score:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[: self.lexical_pool_size]]


def get_document_retriever(settings: RagSettings | None = None) -> BaseRetriever:
    settings = settings or get_rag_settings()
    sync_corpus(settings=settings)

    store = MetadataStore(settings.metadata_db_url)
    store.init_db()
    latest_index = store.get_latest_index()
    if latest_index is None:
        return EmptyRetriever()

    vector_store = load_faiss_index(Path(latest_index.index_path), get_embeddings_model(settings))
    return SourceAwareRetriever(vector_store=vector_store, k=settings.retrieval_k)


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "based",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def _query_terms(query: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9_-]+", query.lower())
        if len(term) > 2 and term not in _STOP_WORDS
    }


def _all_indexed_documents(vector_store: Any) -> list[Document]:
    docstore = getattr(vector_store, "docstore", None)
    documents = getattr(docstore, "_dict", None)
    if isinstance(documents, dict):
        return [doc for doc in documents.values() if isinstance(doc, Document)]
    return []


def _lexical_score(doc: Document, terms: set[str]) -> int:
    content = doc.page_content.lower()
    metadata_text = " ".join(str(value) for value in doc.metadata.values()).lower()
    score = 0
    for term in terms:
        if term in content:
            score += 3
        if term in metadata_text:
            score += 2
    return score


def _quality_score(doc: Document, terms: set[str]) -> float:
    content = doc.page_content
    normalized_content = content.lower()
    score = float(_lexical_score(doc, terms))

    if any(phrase in normalized_content for phrase in ("abstract", "introduction", "called", "we introduce")):
        score += 2.0
    if any(phrase in normalized_content for phrase in ("table ", "figure ", "flops", "argmin")):
        score -= 1.5

    alphanumeric_chars = sum(character.isalnum() for character in content)
    digit_chars = sum(character.isdigit() for character in content)
    if alphanumeric_chars:
        digit_ratio = digit_chars / alphanumeric_chars
        score -= min(digit_ratio * 12, 4)

    return score


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    unique_docs: list[Document] = []
    seen: set[tuple[Any, Any, str]] = set()
    for doc in documents:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page") or doc.metadata.get("row") or doc.metadata.get("record"),
            doc.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs
