import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.settings import RagSettings, get_rag_settings

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    transformers_logging = None


@lru_cache(maxsize=4)
def _cached_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    print("Loading HuggingFace Embeddings...")
    return HuggingFaceEmbeddings(model_name=model_name)


def get_embeddings_model(settings: RagSettings | None = None) -> HuggingFaceEmbeddings:
    settings = settings or get_rag_settings()
    return _cached_embeddings(settings.embedding_model)
