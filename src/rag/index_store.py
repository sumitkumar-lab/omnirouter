from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def build_faiss_index(chunks: list[Document], embeddings: HuggingFaceEmbeddings, index_path: Path) -> Path:
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path.as_posix())
    return index_path


def load_faiss_index(index_path: Path, embeddings: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.load_local(
        index_path.as_posix(),
        embeddings,
        allow_dangerous_deserialization=True,
    )
