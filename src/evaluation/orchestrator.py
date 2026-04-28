from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Callable

from src.evaluation.agent_eval import evaluate_agent_trace
from src.evaluation.dataset import load_evaluation_dataset
from src.evaluation.generation import GenerationJudge, evaluate_generation
from src.evaluation.logging_utils import print_evaluation_summary, save_evaluation_report
from src.evaluation.reranking import evaluate_reranking
from src.evaluation.retrieval import evaluate_retrieval
from src.evaluation.schemas import (
    EvaluationReport,
    EvaluationSummary,
    QueryEvaluationLog,
    QueryExecutionRecord,
)


class EvaluationOrchestrator:
    def __init__(self, generation_judge: GenerationJudge | None = None):
        self.generation_judge = generation_judge

    def evaluate_records(
        self,
        records: list[QueryExecutionRecord],
        retrieval_k: int | None = None,
        output_path: str | Path | None = None,
    ) -> EvaluationReport:
        query_logs: list[QueryEvaluationLog] = []

        for record in records:
            retrieval_metrics = evaluate_retrieval(
                relevant_doc_ids=record.sample.relevant_doc_ids,
                retrieved_docs=record.retrieved_docs,
                k=retrieval_k,
            )
            rerank_metrics = evaluate_reranking(
                relevant_doc_ids=record.sample.relevant_doc_ids,
                before_docs=record.retrieved_docs,
                after_docs=record.reranked_docs or record.retrieved_docs,
            )
            generation_metrics = evaluate_generation(
                answer=record.final_answer,
                ground_truth_answer=record.sample.ground_truth_answer,
                retrieved_context=record.retrieved_context,
                judge=self.generation_judge,
            )
            agent_metrics = evaluate_agent_trace(record.sample, record.trace)

            query_logs.append(
                QueryEvaluationLog(
                    query=record.sample.query,
                    ground_truth_answer=record.sample.ground_truth_answer,
                    relevant_doc_ids=record.sample.relevant_doc_ids,
                    retrieved_docs=[doc.doc_id for doc in record.retrieved_docs],
                    reranked_docs=[doc.doc_id for doc in (record.reranked_docs or record.retrieved_docs)],
                    final_answer=record.final_answer,
                    retrieval_metrics=retrieval_metrics,
                    rerank_metrics=rerank_metrics,
                    generation_metrics=generation_metrics,
                    agent_metrics=agent_metrics,
                )
            )

        report = EvaluationReport(summary=_build_summary(query_logs), query_logs=query_logs)
        if output_path is not None:
            report.report_path = save_evaluation_report(report, output_path)
        print_evaluation_summary(report)
        return report


def run_evaluation_pipeline(
    dataset_path: str | Path,
    record_builder: Callable[[list], list[QueryExecutionRecord]],
    retrieval_k: int | None = None,
    output_path: str | Path | None = None,
    generation_judge: GenerationJudge | None = None,
) -> EvaluationReport:
    samples = load_evaluation_dataset(dataset_path)
    records = record_builder(samples)
    orchestrator = EvaluationOrchestrator(generation_judge=generation_judge)
    return orchestrator.evaluate_records(records, retrieval_k=retrieval_k, output_path=output_path)


def _build_summary(query_logs: list[QueryEvaluationLog]) -> EvaluationSummary:
    if not query_logs:
        return EvaluationSummary(
            total_queries=0,
            avg_recall_at_k=0.0,
            avg_precision_at_k=0.0,
            avg_mrr=0.0,
            avg_ndcg_before=0.0,
            avg_ndcg_after=0.0,
            avg_rank_improvement=0.0,
            avg_semantic_similarity=0.0,
            avg_faithfulness_score=0.0,
            hallucination_rate=0.0,
            avg_tool_usage_score=0.0,
            avg_step_efficiency_score=0.0,
            avg_logical_consistency_score=0.0,
        )

    return EvaluationSummary(
        total_queries=len(query_logs),
        avg_recall_at_k=round(mean(log.retrieval_metrics.recall_at_k for log in query_logs), 4),
        avg_precision_at_k=round(mean(log.retrieval_metrics.precision_at_k for log in query_logs), 4),
        avg_mrr=round(mean(log.retrieval_metrics.reciprocal_rank for log in query_logs), 4),
        avg_ndcg_before=round(mean(log.rerank_metrics.ndcg_before for log in query_logs), 4),
        avg_ndcg_after=round(mean(log.rerank_metrics.ndcg_after for log in query_logs), 4),
        avg_rank_improvement=round(mean(log.rerank_metrics.rank_improvement for log in query_logs), 4),
        avg_semantic_similarity=round(mean(log.generation_metrics.semantic_similarity for log in query_logs), 4),
        avg_faithfulness_score=round(mean(log.generation_metrics.faithfulness_score for log in query_logs), 4),
        hallucination_rate=round(
            mean(1.0 if log.generation_metrics.hallucination_detected else 0.0 for log in query_logs),
            4,
        ),
        avg_tool_usage_score=round(mean(log.agent_metrics.tool_usage_score for log in query_logs), 4),
        avg_step_efficiency_score=round(mean(log.agent_metrics.step_efficiency_score for log in query_logs), 4),
        avg_logical_consistency_score=round(
            mean(log.agent_metrics.logical_consistency_score for log in query_logs),
            4,
        ),
    )
