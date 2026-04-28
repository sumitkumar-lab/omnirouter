from src.evaluation.generation import GenerationJudge, JudgeResult
from src.evaluation.orchestrator import EvaluationOrchestrator, run_evaluation_pipeline
from src.evaluation.schemas import AgentTraceStep, EvaluationSample, QueryExecutionRecord, RetrievedDocument


class StableJudge(GenerationJudge):
    def score(self, answer: str, context: str, ground_truth_answer: str) -> JudgeResult:
        return JudgeResult(faithfulness_score=1.0, hallucination_score=0.0, reasoning="Grounded.")


def test_orchestrator_aggregates_query_metrics_and_writes_report(tmp_path):
    sample = EvaluationSample(
        query="What is OmniRouter?",
        ground_truth_answer="OmniRouter routes LLM requests.",
        relevant_doc_ids=["doc-2"],
        expected_tools=["search_documentation"],
        max_steps=3,
    )
    record = QueryExecutionRecord(
        sample=sample,
        retrieved_docs=[
            RetrievedDocument(doc_id="doc-1", content="noise"),
            RetrievedDocument(doc_id="doc-2", content="OmniRouter routes LLM requests."),
        ],
        reranked_docs=[
            RetrievedDocument(doc_id="doc-2", content="OmniRouter routes LLM requests."),
            RetrievedDocument(doc_id="doc-1", content="noise"),
        ],
        final_answer="OmniRouter routes LLM requests.",
        retrieved_context="OmniRouter routes LLM requests.",
        trace=[AgentTraceStep(step_id=1, kind="tool", tool_name="search_documentation", output_payload={"hits": 1})],
    )

    orchestrator = EvaluationOrchestrator(generation_judge=StableJudge())
    report = orchestrator.evaluate_records([record], retrieval_k=2, output_path=tmp_path / "final_report.json")

    assert report.summary.total_queries == 1
    assert report.summary.avg_recall_at_k == 1.0
    assert report.summary.avg_mrr == 0.5
    assert report.report_path == tmp_path / "final_report.json"
    assert report.report_path.exists()


def test_run_evaluation_pipeline_executes_builder_and_returns_report(tmp_path):
    def record_builder(samples):
        sample = samples[0]
        return [
            QueryExecutionRecord(
                sample=sample,
                retrieved_docs=[RetrievedDocument(doc_id="doc-1", content="OmniRouter routes LLM requests.")],
                reranked_docs=[RetrievedDocument(doc_id="doc-1", content="OmniRouter routes LLM requests.")],
                final_answer=sample.ground_truth_answer,
                retrieved_context=sample.ground_truth_answer,
                trace=[],
            )
        ]

    report = run_evaluation_pipeline(
        dataset_path="tests/fixtures/evaluation_dataset.json",
        record_builder=record_builder,
        retrieval_k=1,
        output_path=tmp_path / "pipeline_report.json",
        generation_judge=StableJudge(),
    )

    assert report.summary.total_queries == 1
    assert report.report_path == tmp_path / "pipeline_report.json"
