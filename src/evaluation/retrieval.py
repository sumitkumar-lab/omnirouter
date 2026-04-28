from __future__ import annotations

from src.evaluation.schemas import RetrievedDocument, RetrievalMetrics


def evaluate_retrieval(
    relevant_doc_ids: list[str],
    retrieved_docs: list[RetrievedDocument],
    k: int | None = None,
) -> RetrievalMetrics:
    if k is None:
        k = len(retrieved_docs)

    relevant_set = set(relevant_doc_ids)
    top_docs = retrieved_docs[:k]
    top_doc_ids = [doc.doc_id for doc in top_docs]
    hits = sum(1 for doc_id in top_doc_ids if doc_id in relevant_set)

    recall = hits / len(relevant_set) if relevant_set else 0.0
    precision = hits / len(top_doc_ids) if top_doc_ids else 0.0
    reciprocal_rank = 0.0

    for index, doc_id in enumerate(top_doc_ids, start=1):
        if doc_id in relevant_set:
            reciprocal_rank = 1.0 / index
            break

    return RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
        relevant_hits=hits,
    )
