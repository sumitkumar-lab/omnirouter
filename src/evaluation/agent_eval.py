from __future__ import annotations

from src.evaluation.schemas import AgentMetrics, AgentTraceStep, EvaluationSample


def evaluate_agent_trace(sample: EvaluationSample, trace: list[AgentTraceStep]) -> AgentMetrics:
    tool_steps = [step for step in trace if step.tool_name]
    used_tools = [step.tool_name for step in tool_steps if step.tool_name]
    tool_usage_score = _tool_usage_score(sample.expected_tools, used_tools)
    step_efficiency_score = _step_efficiency_score(sample.max_steps, len(trace))
    logical_consistency_score = _logical_consistency_score(trace)

    return AgentMetrics(
        tool_usage_score=tool_usage_score,
        step_efficiency_score=step_efficiency_score,
        logical_consistency_score=logical_consistency_score,
        total_steps=len(trace),
        used_tools=used_tools,
    )


def _tool_usage_score(expected_tools: list[str], used_tools: list[str]) -> float:
    if not expected_tools:
        return 1.0
    if not used_tools:
        return 0.0
    expected_set = set(expected_tools)
    used_set = set(used_tools)
    return round(len(expected_set & used_set) / len(expected_set), 4)


def _step_efficiency_score(max_steps: int | None, total_steps: int) -> float:
    if max_steps is None or max_steps <= 0:
        return 1.0
    if total_steps <= max_steps:
        return 1.0
    overflow = total_steps - max_steps
    return round(max(0.0, 1.0 - (overflow / max_steps)), 4)


def _logical_consistency_score(trace: list[AgentTraceStep]) -> float:
    if not trace:
        return 1.0

    penalty = 0.0
    for step in trace:
        if step.tool_name and not step.output_payload and not step.content:
            penalty += 0.25
        if step.kind == "tool" and not step.tool_name:
            penalty += 0.25

    return round(max(0.0, 1.0 - penalty), 4)
