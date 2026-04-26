from src.rag.retrieval import get_document_retriever
from src.rag.settings import RagSettings


def test_retriever_reads_from_synced_faiss_index(tmp_path):
    raw_dir = tmp_path / "data_lake" / "ops"
    raw_dir.mkdir(parents=True)
    (raw_dir / "incident.txt").write_text(
        "Project Lantern is the escalation channel for documentation incidents.",
        encoding="utf-8",
    )

    settings = RagSettings(project_root=tmp_path, retrieval_k=2)
    retriever = get_document_retriever(settings)
    docs = retriever.invoke("What is the escalation channel?")

    assert docs
    assert "Project Lantern" in docs[0].page_content
