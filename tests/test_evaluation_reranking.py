from src.evaluation.reranking import evaluate_reranking
from src.evaluation.schemas import RetrievedDocument


def test_evaluate_reranking_detects_relevant_docs_moving_higher():
    before = [
        RetrievedDocument(doc_id="doc-a"),
        RetrievedDocument(doc_id="doc-b"),
        RetrievedDocument(doc_id="doc-c"),
    ]
    after = [
        RetrievedDocument(doc_id="doc-b"),
        RetrievedDocument(doc_id="doc-a"),
        RetrievedDocument(doc_id="doc-c"),
    ]

    metrics = evaluate_reranking(
        relevant_doc_ids=["doc-b"],
        before_docs=before,
        after_docs=after,
    )

    assert metrics.ndcg_after >= metrics.ndcg_before
    assert metrics.rank_improvement == 1.0
    assert metrics.moved_relevant_higher is True
