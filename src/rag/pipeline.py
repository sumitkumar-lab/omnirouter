from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from src.rag.embeddings import get_embeddings_model
from src.rag.index_store import build_faiss_index
from src.rag.ingestion import build_chunks_from_sources, compute_source_manifest, discover_document_paths, ensure_data_lake_layout
from src.rag.metadata_store import MetadataStore
from src.rag.settings import RagSettings, get_rag_settings

@dataclass(slots=True)
class SyncResult:
    rebuilt: bool
    version_label: str | None
    source_count: int
    chunk_count: int
    index_path: Path | None


def sync_corpus(settings: RagSettings | None = None, force: bool = False) -> SyncResult:
    settings = settings or get_rag_settings()
    ensure_data_lake_layout(settings)

    store = MetadataStore(settings.metadata_db_url)
    store.init_db()

    document_paths = discover_document_paths(settings)
    manifest = compute_source_manifest(document_paths, settings)
    latest = store.get_latest_corpus_version()

    latest_index = store.get_latest_index()
    latest_index_path = Path(latest_index.index_path) if latest_index is not None else None

    if (
        not force
        and latest is not None
        and latest.manifest == manifest
        and (latest_index_path is None or latest_index_path.exists())
    ):
        return SyncResult(
            rebuilt=False,
            version_label=latest.version_label,
            source_count=latest.source_count,
            chunk_count=latest.chunk_count,
            index_path=latest_index_path,
        )

    chunks, manifest = build_chunks_from_sources(settings)
    version_label = _next_version_label(latest.version_label if latest else None, settings)
    version_dir = settings.corpus_dir / version_label
    index_dir = version_dir / "faiss_index"

    version_dir.mkdir(parents=True, exist_ok=True)
    _write_version_snapshot(version_dir, manifest, chunks)

    index_path = None
    if chunks:
        embeddings = get_embeddings_model(settings)
        index_path = build_faiss_index(chunks, embeddings, index_dir)

    store.create_corpus_version(
        version_label=version_label,
        manifest=manifest,
        source_count=len(manifest),
        chunk_count=len(chunks),
        index_path=index_path,
        embedding_model=settings.embedding_model,
    )

    return SyncResult(
        rebuilt=True,
        version_label=version_label,
        source_count=len(manifest),
        chunk_count=len(chunks),
        index_path=index_path,
    )


def _next_version_label(current: str | None, settings: RagSettings) -> str:
    if current is None:
        return f"{settings.version_prefix}1"
    try:
        version_number = int(current.removeprefix(settings.version_prefix))
    except ValueError:
        version_number = 0
    return f"{settings.version_prefix}{version_number + 1}"


def _write_version_snapshot(
    version_dir: Path,
    manifest: dict[str, dict[str, object]],
    chunks: list[Document],
) -> None:
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (version_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps({"page_content": chunk.page_content, "metadata": chunk.metadata}, ensure_ascii=True) + "\n"
            )
