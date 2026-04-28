from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.schemas import RetrievedDocument


def test_evaluate_retrieval_computes_recall_precision_and_mrr():
    metrics = evaluate_retrieval(
        relevant_doc_ids=["doc-2", "doc-4"],
        retrieved_docs=[
            RetrievedDocument(doc_id="doc-1"),
            RetrievedDocument(doc_id="doc-2"),
            RetrievedDocument(doc_id="doc-3"),
        ],
        k=3,
    )

    assert metrics.recall_at_k == 0.5
    assert round(metrics.precision_at_k, 4) == 0.3333
    assert metrics.reciprocal_rank == 0.5
