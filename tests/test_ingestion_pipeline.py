import json

from src.rag.ingestion import build_chunks_from_sources, compute_source_manifest, discover_document_paths, load_documents
from src.rag.settings import RagSettings


def test_discover_and_load_multi_format_documents(tmp_path):
    (tmp_path / "data_lake").mkdir(parents=True)
    (tmp_path / "data_lake" / "guide.txt").write_text("Router failover keeps production alive.", encoding="utf-8")
    (tmp_path / "data_lake" / "notes.md").write_text("# Notes\nRAG supports markdown too.", encoding="utf-8")
    (tmp_path / "data_lake" / "sales.csv").write_text("name,revenue\nAda,100\nLinus,200\n", encoding="utf-8")
    (tmp_path / "data_lake" / "config.json").write_text(
        json.dumps({"provider": "openai", "fallback": "anthropic"}),
        encoding="utf-8",
    )

    settings = RagSettings(project_root=tmp_path)

    paths = discover_document_paths(settings)
    documents = load_documents(paths, settings)
    chunks, manifest = build_chunks_from_sources(settings)

    assert {path.name for path in paths} == {"guide.txt", "notes.md", "sales.csv", "config.json"}
    assert manifest == compute_source_manifest(paths, settings)
    assert any(doc.metadata["file_type"] == "txt" for doc in documents)
    assert any(doc.metadata["file_type"] == "md" for doc in documents)
    assert any(doc.metadata["file_type"] == "csv" for doc in documents)
    assert any(doc.metadata["file_type"] == "json" for doc in documents)
    assert any(doc.metadata["source_name"] == "guide" for doc in documents)
    assert chunks


def test_pdf_prefers_markdown_sidecar_and_chunks_by_headers(tmp_path):
    data_lake = tmp_path / "data_lake"
    data_lake.mkdir(parents=True)
    pdf_path = data_lake / "chinchilla_train.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    pdf_path.with_suffix(".md").write_text(
        "# Chinchilla\n\n"
        "Authors: DeepMind\n\n"
        "## Methodology\n\n"
        "We train Chinchilla with compute-optimal scaling.\n\n"
        "$$L(N, D) = E + A/N^a + B/D^b$$\n\n"
        "## Proofs\n\n"
        "The derivation keeps model size and training tokens balanced.",
        encoding="utf-8",
    )

    settings = RagSettings(project_root=tmp_path)
    documents = load_documents([pdf_path], settings)

    assert {document.metadata["chunk_header"] for document in documents} >= {"Chinchilla", "Methodology", "Proofs"}
    assert all(document.metadata["source"] == "data_lake/chinchilla_train.pdf" for document in documents)
    assert all(document.metadata["content_type"] == "ocr_markdown" for document in documents)
    assert any("$$L(N, D)" in document.page_content for document in documents)
