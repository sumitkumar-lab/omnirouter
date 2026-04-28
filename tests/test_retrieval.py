from langchain_core.documents import Document

from src.rag.retrieval import SourceAwareRetriever
from src.rag.retrieval import get_document_retriever
from src.rag.settings import RagSettings


class FakeDocstore:
    def __init__(self, documents):
        self._dict = {str(index): document for index, document in enumerate(documents)}


class FakeVectorStore:
    def __init__(self, documents, semantic_results):
        self.docstore = FakeDocstore(documents)
        self.semantic_results = semantic_results

    def similarity_search(self, query, k=4):
        return self.semantic_results[:k]


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


def test_source_aware_retriever_prioritizes_exact_document_terms():
    omnirouter_doc = Document(
        page_content="Omnirouter is a router engine.",
        metadata={"source": "data_lake/source_name/omnirouter_facts.txt"},
    )
    chinchilla_doc = Document(
        page_content="Chinchilla training uses compute-optimal scaling formulas.",
        metadata={"source": "data_lake/source_name/chinchilla train.pdf"},
    )
    vector_store = FakeVectorStore(
        documents=[omnirouter_doc, chinchilla_doc],
        semantic_results=[chinchilla_doc, omnirouter_doc],
    )

    retriever = SourceAwareRetriever(vector_store=vector_store, k=2)
    docs = retriever.invoke("What is omnirouter?")

    assert docs[0] == omnirouter_doc


def test_source_aware_retriever_demotes_numeric_chart_noise():
    explanatory_doc = Document(
        page_content=(
            "We introduce Chinchilla, a compute-optimal large language model "
            "with 70B parameters trained on 1.4 trillion tokens."
        ),
        metadata={"source": "data_lake/chinchilla train.pdf"},
    )
    chart_doc = Document(
        page_content=(
            "1017 1019 1021 1023 1025 FLOPs 10M 100M 1.0B 10B 100B 1T "
            "Parameters Figure 1 Chinchilla Gopher GPT-3"
        ),
        metadata={"source": "data_lake/chinchilla train.pdf"},
    )
    vector_store = FakeVectorStore(
        documents=[explanatory_doc, chart_doc],
        semantic_results=[chart_doc, explanatory_doc],
    )

    retriever = SourceAwareRetriever(vector_store=vector_store, k=2)
    docs = retriever.invoke("What is chinchilla?")

    assert docs[0] == explanatory_doc
