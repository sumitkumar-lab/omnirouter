import json

from src.rag.ingestion import build_chunks_from_sources, compute_source_manifest, discover_document_paths, load_documents
from src.rag.settings import RagSettings


def test_discover_and_load_multi_format_documents(tmp_path):
    (tmp_path / "data_lake" / "engineering").mkdir(parents=True)
    (tmp_path / "data_lake" / "engineering" / "guide.txt").write_text("Router failover keeps production alive.", encoding="utf-8")
    (tmp_path / "data_lake" / "engineering" / "notes.md").write_text("# Notes\nRAG supports markdown too.", encoding="utf-8")
    (tmp_path / "data_lake" / "engineering" / "sales.csv").write_text("name,revenue\nAda,100\nLinus,200\n", encoding="utf-8")
    (tmp_path / "data_lake" / "engineering" / "config.json").write_text(
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
    assert chunks
