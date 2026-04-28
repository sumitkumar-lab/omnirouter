from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    query: str
    ground_truth_answer: str
    relevant_doc_ids: list[str]
    expected_tools: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    doc_id: str
    content: str = ""
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTraceStep(BaseModel):
    step_id: int
    kind: str = "reasoning"
    content: str = ""
    tool_name: str | None = None
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)


class QueryExecutionRecord(BaseModel):
    sample: EvaluationSample
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)
    reranked_docs: list[RetrievedDocument] = Field(default_factory=list)
    final_answer: str = ""
    retrieved_context: str = ""
    trace: list[AgentTraceStep] = Field(default_factory=list)


class RetrievalMetrics(BaseModel):
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    relevant_hits: int


class RerankMetrics(BaseModel):
    ndcg_before: float
    ndcg_after: float
    rank_improvement: float
    moved_relevant_higher: bool


class GenerationMetrics(BaseModel):
    semantic_similarity: float
    faithfulness_score: float
    hallucination_score: float
    hallucination_detected: bool
    judge_reasoning: str = ""


class AgentMetrics(BaseModel):
    tool_usage_score: float
    step_efficiency_score: float
    logical_consistency_score: float
    total_steps: int
    used_tools: list[str] = Field(default_factory=list)


class QueryEvaluationLog(BaseModel):
    query: str
    ground_truth_answer: str
    relevant_doc_ids: list[str]
    retrieved_docs: list[str]
    reranked_docs: list[str]
    final_answer: str
    retrieval_metrics: RetrievalMetrics
    rerank_metrics: RerankMetrics
    generation_metrics: GenerationMetrics
    agent_metrics: AgentMetrics


class EvaluationSummary(BaseModel):
    total_queries: int
    avg_recall_at_k: float
    avg_precision_at_k: float
    avg_mrr: float
    avg_ndcg_before: float
    avg_ndcg_after: float
    avg_rank_improvement: float
    avg_semantic_similarity: float
    avg_faithfulness_score: float
    hallucination_rate: float
    avg_tool_usage_score: float
    avg_step_efficiency_score: float
    avg_logical_consistency_score: float


class EvaluationReport(BaseModel):
    summary: EvaluationSummary
    query_logs: list[QueryEvaluationLog]
    report_path: Path | None = None
