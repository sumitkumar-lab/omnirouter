import json

from src.evaluation.logging_utils import save_evaluation_report
from src.evaluation.schemas import (
    AgentMetrics,
    EvaluationReport,
    EvaluationSummary,
    GenerationMetrics,
    QueryEvaluationLog,
    RetrievalMetrics,
    RerankMetrics,
)


def test_save_evaluation_report_writes_structured_json(tmp_path):
    report = EvaluationReport(
        summary=EvaluationSummary(
            total_queries=1,
            avg_recall_at_k=1.0,
            avg_precision_at_k=1.0,
            avg_mrr=1.0,
            avg_ndcg_before=0.5,
            avg_ndcg_after=1.0,
            avg_rank_improvement=1.0,
            avg_semantic_similarity=0.9,
            avg_faithfulness_score=0.8,
            hallucination_rate=0.0,
            avg_tool_usage_score=1.0,
            avg_step_efficiency_score=1.0,
            avg_logical_consistency_score=1.0,
        ),
        query_logs=[
            QueryEvaluationLog(
                query="q",
                ground_truth_answer="a",
                relevant_doc_ids=["doc-1"],
                retrieved_docs=["doc-1"],
                reranked_docs=["doc-1"],
                final_answer="ans",
                retrieval_metrics=RetrievalMetrics(
                    recall_at_k=1.0,
                    precision_at_k=1.0,
                    reciprocal_rank=1.0,
                    relevant_hits=1,
                ),
                rerank_metrics=RerankMetrics(
                    ndcg_before=0.5,
                    ndcg_after=1.0,
                    rank_improvement=1.0,
                    moved_relevant_higher=True,
                ),
                generation_metrics=GenerationMetrics(
                    semantic_similarity=0.9,
                    faithfulness_score=0.8,
                    hallucination_score=0.0,
                    hallucination_detected=False,
                    judge_reasoning="Grounded.",
                ),
                agent_metrics=AgentMetrics(
                    tool_usage_score=1.0,
                    step_efficiency_score=1.0,
                    logical_consistency_score=1.0,
                    total_steps=2,
                    used_tools=["search_documentation"],
                ),
            )
        ],
    )

    output_path = tmp_path / "report.json"
    saved_path = save_evaluation_report(report, output_path)
    payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == output_path
    assert payload["summary"]["avg_mrr"] == 1.0
    assert payload["query_logs"][0]["retrieved_docs"] == ["doc-1"]
