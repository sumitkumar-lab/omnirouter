from src.evaluation.agent_eval import evaluate_agent_trace
from src.evaluation.schemas import AgentTraceStep, EvaluationSample


def test_evaluate_agent_trace_scores_tools_efficiency_and_logic():
    sample = EvaluationSample(
        query="What is OmniRouter?",
        ground_truth_answer="OmniRouter routes requests.",
        relevant_doc_ids=["doc-1"],
        expected_tools=["search_documentation"],
        max_steps=3,
    )
    trace = [
        AgentTraceStep(step_id=1, kind="reasoning", content="Need docs."),
        AgentTraceStep(step_id=2, kind="tool", tool_name="search_documentation", output_payload={"docs": 2}),
        AgentTraceStep(step_id=3, kind="reasoning", content="Answer with evidence."),
    ]

    metrics = evaluate_agent_trace(sample, trace)

    assert metrics.tool_usage_score == 1.0
    assert metrics.step_efficiency_score == 1.0
    assert metrics.logical_consistency_score == 1.0
