import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RagSettings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_lake_dirname: str = "data_lake"
    corpus_dirname: str = "corpus"
    metadata_db_filename: str = "rag_metadata.db"
    metadata_db_url: str | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    version_prefix: str = "version_v"
    supported_extensions: tuple[str, ...] = (".pdf", ".txt", ".md", ".csv", ".json")

    def __post_init__(self) -> None:
        if self.metadata_db_url is None:
            env_db_url = os.getenv("RAG_METADATA_DB_URL")
            if env_db_url:
                self.metadata_db_url = env_db_url
            else:
                self.metadata_db_url = f"sqlite:///{(self.project_root / self.metadata_db_filename).as_posix()}"

    @property
    def data_lake_dir(self) -> Path:
        return self.project_root / self.data_lake_dirname

    @property
    def corpus_dir(self) -> Path:
        return self.project_root / self.corpus_dirname


def get_rag_settings() -> RagSettings:
    return RagSettings()
