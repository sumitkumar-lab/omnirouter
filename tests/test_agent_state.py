from langchain_core.messages import HumanMessage

from src.agent.graph import _build_research_plan, _route_from_messages, _worker_prompt
from src.agent.state import ResearchState


def test_orchestrator_routes_and_builds_research_plan_without_message_pollution():
    state: ResearchState = {
        "messages": [HumanMessage(content="Derive the gradient of the Chinchilla loss equation.")],
        "literature_notes": ["[Literature Reviewer]\nChinchilla uses compute-optimal scaling."],
        "math_scratchpad": [],
        "code_scratchpad": [],
    }

    route = _route_from_messages(state["messages"])
    plan = _build_research_plan(route, state)
    prompt = _worker_prompt("Base mathematician prompt.", {**state, "research_plan": plan})

    assert route == "mathematician"
    assert "math_scratchpad" in plan
    assert "Chinchilla uses compute-optimal scaling" in prompt
    assert len(state["messages"]) == 1
