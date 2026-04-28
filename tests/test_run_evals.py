from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.evaluation.run_evals import build_execution_record, evaluate_real_agent, extract_evaluation_payload
from src.evaluation.schemas import AgentTraceStep, EvaluationSample, RetrievedDocument


def test_extract_evaluation_payload_collects_tool_context_and_final_answer():
    final_state = {
        "messages": [
            HumanMessage(content="question"),
            ToolMessage(content="doc chunk 1", tool_call_id="1"),
            ToolMessage(content="doc chunk 2", tool_call_id="2"),
            AIMessage(content="Final answer"),
        ]
    }

    context, answer = extract_evaluation_payload(final_state)

    assert context == "doc chunk 1\ndoc chunk 2\n"
    assert answer == "Final answer"


def test_build_execution_record_maps_agent_output_to_query_record():
    sample = EvaluationSample(
        query="What is OmniRouter?",
        ground_truth_answer="OmniRouter routes LLM requests.",
        relevant_doc_ids=["doc-1"],
    )
    final_state = {
        "messages": [
            ToolMessage(content="Router context", tool_call_id="1"),
            AIMessage(content="OmniRouter routes LLM requests."),
        ]
    }

    record = build_execution_record(
        sample=sample,
        final_state=final_state,
        retrieved_docs=[RetrievedDocument(doc_id="doc-1", content="Router context")],
        trace=[AgentTraceStep(step_id=1, kind="tool", tool_name="search_documentation")],
    )

    assert record.retrieved_context == "Router context\n"
    assert record.final_answer == "OmniRouter routes LLM requests."
    assert record.retrieved_docs[0].doc_id == "doc-1"


def test_evaluate_real_agent_runs_through_orchestrator():
    class FakeAgent:
        def invoke(self, initial_state, config):
            return {
                "messages": [
                    ToolMessage(content="OmniRouter routes requests.", tool_call_id="1"),
                    AIMessage(content="OmniRouter routes requests."),
                ]
            }

    report = evaluate_real_agent(
        query="What is OmniRouter?",
        relevant_doc_ids=["tool_doc_1"],
        ground_truth_answer="OmniRouter routes requests.",
        agent_app=FakeAgent(),
        thread_id="eval-thread",
    )

    assert report.summary.total_queries == 1
    assert report.query_logs[0].retrieved_docs == ["tool_doc_1"]
