from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.settings import RagSettings, get_rag_settings

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def ensure_data_lake_layout(settings: RagSettings | None = None) -> None:
    settings = settings or get_rag_settings()
    settings.data_lake_dir.mkdir(parents=True, exist_ok=True)
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)


def chunk_document_text(
    raw_text: str,
    source: str = "data_lake/inline/source.txt",
    settings: RagSettings | None = None,
) -> list[Document]:
    return chunk_documents([Document(page_content=raw_text, metadata={"source": source})], settings=settings)


def chunk_documents(documents: list[Document], settings: RagSettings | None = None) -> list[Document]:
    settings = settings or get_rag_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def discover_document_paths(settings: RagSettings | None = None) -> list[Path]:
    settings = settings or get_rag_settings()
    ensure_data_lake_layout(settings)

    paths: list[Path] = []
    for path in settings.data_lake_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in settings.supported_extensions:
            paths.append(path.resolve())
    return sorted(paths)


def compute_source_manifest(
    document_paths: Iterable[Path],
    settings: RagSettings | None = None,
) -> dict[str, dict[str, Any]]:
    settings = settings or get_rag_settings()
    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(document_paths):
        raw_bytes = path.read_bytes()
        relative_path = path.resolve().relative_to(settings.project_root).as_posix()
        stat = path.stat()
        manifest[relative_path] = {
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return manifest


def load_documents(
    document_paths: Iterable[Path],
    settings: RagSettings | None = None,
) -> list[Document]:
    settings = settings or get_rag_settings()
    documents: list[Document] = []

    for path in document_paths:
        suffix = path.suffix.lower()
        if suffix == ".md":
            documents.extend(_load_markdown_document(path, settings))
        elif suffix == ".txt":
            documents.extend(_load_text_document(path, settings))
        elif suffix == ".csv":
            documents.extend(_load_csv_document(path, settings))
        elif suffix == ".json":
            documents.extend(_load_json_document(path, settings))
        elif suffix == ".pdf":
            documents.extend(_load_pdf_document(path, settings))
    return documents


def build_chunks_from_sources(
    settings: RagSettings | None = None,
) -> tuple[list[Document], dict[str, dict[str, Any]]]:
    settings = settings or get_rag_settings()
    document_paths = discover_document_paths(settings)
    manifest = compute_source_manifest(document_paths, settings)
    documents = load_documents(document_paths, settings)
    return chunk_documents(documents, settings), manifest


def _relative_source(path: Path, settings: RagSettings) -> str:
    return path.resolve().relative_to(settings.project_root).as_posix()


def _source_name(path: Path, settings: RagSettings) -> str:
    relative_path = path.resolve().relative_to(settings.data_lake_dir.resolve())
    if len(relative_path.parts) <= 1:
        return path.stem
    return relative_path.parts[0]


def _base_metadata(path: Path, settings: RagSettings) -> dict[str, str]:
    return {
        "source": _relative_source(path, settings),
        "source_name": _source_name(path, settings),
        "file_name": path.name,
        "document_name": path.stem,
        "file_type": path.suffix.lower().lstrip("."),
    }


def _metadata_with_overrides(path: Path, settings: RagSettings, **overrides: Any) -> dict[str, Any]:
    metadata = _base_metadata(path, settings)
    metadata.update(overrides)
    return metadata


def _load_text_document(path: Path, settings: RagSettings) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        Document(
            page_content=text,
            metadata=_base_metadata(path, settings),
        )
    ]


def _load_markdown_document(path: Path, settings: RagSettings) -> list[Document]:
    markdown = path.read_text(encoding="utf-8", errors="replace")
    return _section_documents_from_markdown(
        markdown=markdown,
        source_path=path,
        settings=settings,
        metadata_overrides={"file_type": "md", "content_type": "markdown"},
    )


def _load_csv_document(path: Path, settings: RagSettings) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False

        if has_header:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=1):
                documents.append(
                    Document(
                        page_content="\n".join(f"{key}: {value}" for key, value in row.items()),
                        metadata={
                            **_metadata_with_overrides(path, settings),
                            "row": row_number,
                        },
                    )
                )
        else:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                documents.append(
                    Document(
                        page_content=", ".join(row),
                        metadata={
                            **_metadata_with_overrides(path, settings),
                            "row": row_number,
                        },
                    )
                )
    return documents


def _load_json_document(path: Path, settings: RagSettings) -> list[Document]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, list):
        return [
            Document(
                page_content=json.dumps(item, indent=2, sort_keys=True, ensure_ascii=True),
                metadata={
                    **_metadata_with_overrides(path, settings),
                    "record": index,
                },
            )
            for index, item in enumerate(data, start=1)
        ]

    return [
        Document(
            page_content=json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True),
            metadata=_base_metadata(path, settings),
        )
    ]


def _load_pdf_document(path: Path, settings: RagSettings) -> list[Document]:
    sidecar = _markdown_sidecar_for(path)
    if sidecar is not None:
        markdown = sidecar.read_text(encoding="utf-8", errors="replace")
        return _section_documents_from_markdown(
            markdown=markdown,
            source_path=path,
            settings=settings,
            metadata_overrides={
                "file_type": "pdf",
                "content_type": "ocr_markdown",
                "ocr_sidecar": _relative_source(sidecar, settings),
            },
        )

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        documents.append(
            Document(
                page_content=page_text,
                metadata={
                    **_metadata_with_overrides(path, settings, content_type="pdf_text"),
                    "page": page_number,
                },
            )
        )
    return documents


def _markdown_sidecar_for(path: Path) -> Path | None:
    for suffix in (".md", ".markdown"):
        candidate = path.with_suffix(suffix)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _section_documents_from_markdown(
    markdown: str,
    source_path: Path,
    settings: RagSettings,
    metadata_overrides: dict[str, Any],
) -> list[Document]:
    sections = _split_markdown_by_headers(markdown)
    documents: list[Document] = []
    for index, section in enumerate(sections, start=1):
        documents.append(
            Document(
                page_content=section["text"],
                metadata=_metadata_with_overrides(
                    source_path,
                    settings,
                    **metadata_overrides,
                    chunk_header=section["header"],
                    section_path=section["section_path"],
                    section=index,
                    chunking_strategy="markdown_headers",
                ),
            )
        )
    return documents


def _split_markdown_by_headers(markdown: str) -> list[dict[str, Any]]:
    matches = list(HEADER_RE.finditer(markdown))
    if not matches:
        text = markdown.strip()
        return [{"header": "Document", "section_path": ["Document"], "text": text}] if text else []

    sections: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    if matches[0].start() > 0 and markdown[: matches[0].start()].strip():
        sections.append(
            {
                "header": "Front Matter",
                "section_path": ["Front Matter"],
                "text": markdown[: matches[0].start()].strip(),
            }
        )

    for index, match in enumerate(matches):
        level = len(match.group(1))
        header = _clean_header(match.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, header))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        text = markdown[match.start() : end].strip()
        if text:
            sections.append(
                {
                    "header": header,
                    "section_path": [item[1] for item in stack],
                    "text": text,
                }
            )
    return sections


def _clean_header(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" #\t\r\n")
