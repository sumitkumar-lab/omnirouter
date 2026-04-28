from __future__ import annotations

import math

from src.evaluation.schemas import RetrievedDocument, RerankMetrics


def evaluate_reranking(
    relevant_doc_ids: list[str],
    before_docs: list[RetrievedDocument],
    after_docs: list[RetrievedDocument],
) -> RerankMetrics:
    relevant_set = set(relevant_doc_ids)
    ndcg_before = _ndcg(relevant_set, before_docs)
    ndcg_after = _ndcg(relevant_set, after_docs)
    before_best = _best_rank(relevant_set, before_docs)
    after_best = _best_rank(relevant_set, after_docs)
    rank_improvement = float(before_best - after_best) if before_best and after_best else 0.0

    return RerankMetrics(
        ndcg_before=ndcg_before,
        ndcg_after=ndcg_after,
        rank_improvement=rank_improvement,
        moved_relevant_higher=after_best < before_best if before_best and after_best else False,
    )


def _ndcg(relevant_set: set[str], docs: list[RetrievedDocument]) -> float:
    if not relevant_set or not docs:
        return 0.0

    dcg = 0.0
    for rank, doc in enumerate(docs, start=1):
        relevance = 1.0 if doc.doc_id in relevant_set else 0.0
        if relevance:
            dcg += relevance / math.log2(rank + 1)

    ideal_hits = min(len(relevant_set), len(docs))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _best_rank(relevant_set: set[str], docs: list[RetrievedDocument]) -> int | None:
    for rank, doc in enumerate(docs, start=1):
        if doc.doc_id in relevant_set:
            return rank
    return None
