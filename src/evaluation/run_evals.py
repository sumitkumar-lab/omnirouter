from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.evaluation.orchestrator import EvaluationOrchestrator
from src.evaluation.schemas import AgentTraceStep, EvaluationSample, QueryExecutionRecord, RetrievedDocument


def extract_evaluation_payload(final_state: dict) -> tuple[str, str]:
    retrieved_context = ""
    final_answer = ""

    for msg in final_state["messages"]:
        if isinstance(msg, ToolMessage):
            retrieved_context += msg.content + "\n"
        elif isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content

    return retrieved_context, final_answer


def build_execution_record(
    sample: EvaluationSample,
    final_state: dict,
    retrieved_docs: list[RetrievedDocument],
    reranked_docs: list[RetrievedDocument] | None = None,
    trace: list[AgentTraceStep] | None = None,
) -> QueryExecutionRecord:
    retrieved_context, final_answer = extract_evaluation_payload(final_state)
    return QueryExecutionRecord(
        sample=sample,
        retrieved_docs=retrieved_docs,
        reranked_docs=reranked_docs or retrieved_docs,
        final_answer=final_answer,
        retrieved_context=retrieved_context,
        trace=trace or [],
    )


def evaluate_real_agent(
    query: str,
    relevant_doc_ids: list[str] | None = None,
    ground_truth_answer: str = "",
    agent_app=None,
    orchestrator: EvaluationOrchestrator | None = None,
    thread_id: str = "automated_eval_run_1",
):
    if agent_app is None:
        from src.agent.graph import app as default_app

        agent_app = default_app

    sample = EvaluationSample(
        query=query,
        ground_truth_answer=ground_truth_answer,
        relevant_doc_ids=relevant_doc_ids or [],
    )
    initial_state = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}}
    final_state = agent_app.invoke(initial_state, config)
    retrieved_context, final_answer = extract_evaluation_payload(final_state)

    tool_messages = [msg for msg in final_state["messages"] if isinstance(msg, ToolMessage)]
    retrieved_docs = [
        RetrievedDocument(doc_id=f"tool_doc_{index}", content=msg.content)
        for index, msg in enumerate(tool_messages, start=1)
    ]
    trace = [
        AgentTraceStep(step_id=index, kind="tool" if isinstance(msg, ToolMessage) else "reasoning", content=str(msg))
        for index, msg in enumerate(final_state["messages"], start=1)
    ]

    record = QueryExecutionRecord(
        sample=sample,
        retrieved_docs=retrieved_docs,
        reranked_docs=retrieved_docs,
        final_answer=final_answer,
        retrieved_context=retrieved_context,
        trace=trace,
    )

    orchestrator = orchestrator or EvaluationOrchestrator()
    return orchestrator.evaluate_records([record], output_path=None)


if __name__ == "__main__":
    report = evaluate_real_agent(
        query="What is OmniRouter and what does it do?",
        ground_truth_answer="OmniRouter routes LLM requests and supports retrieval workflows.",
    )
    print(report.summary.model_dump())
