from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.state import ResearchState, WorkerName
from src.agent.tools import (
    execute_python_code,
    generate_pytorch_ablation_script,
    query_postgres_metadata,
    query_research_corpus,
    query_research_metadata,
    search_documentation,
    search_web,
)

load_dotenv()

RouteName = Literal["literature_reviewer", "mathematician", "experimentalist"]

ORCHESTRATOR_PROMPT = (
    "You are the Orchestrator for a single autonomous AI Research Scientist. "
    "Route each user request to exactly one specialist: Literature Reviewer for paper retrieval, "
    "citations, methods, and uploaded corpus questions; Mathematician for derivations, symbolic "
    "checks, and numerical verification; Experimentalist for PyTorch ablations and experiment plans. "
    "Do not answer the research question yourself."
)

LITERATURE_REVIEWER_PROMPT = (
    "You are the Literature Reviewer Node of an AI Research Scientist. "
    "Use the local FAISS research corpus first. Use metadata when source identity, corpus version, "
    "or indexed papers matter. Use DuckDuckGo only when local papers do not contain enough context "
    "or the user asks for external/current literature. Answer with cited source names when available."
)

MATHEMATICIAN_PROMPT = (
    "You are the Mathematician Node of an AI Research Scientist. "
    "Translate derivation questions into precise symbolic or numerical checks. "
    "Use the secure Python tool with sympy and numpy for nontrivial algebra, optimization, matrices, "
    "or sanity checks. Explain the result briefly after tool output is available."
)

EXPERIMENTALIST_PROMPT = (
    "You are the Experimentalist Node of an AI Research Scientist. "
    "Produce raw PyTorch ablation scripts or concise experiment designs. "
    "When a script is requested, output runnable code with minimal commentary. "
    "Use the local corpus when the ablation should match an uploaded paper."
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

literature_tools = [
    search_documentation,
    query_research_corpus,
    query_research_metadata,
    query_postgres_metadata,
    search_web,
]
mathematician_tools = [execute_python_code]
experimentalist_tools = [generate_pytorch_ablation_script, search_documentation]
all_tools = [
    search_documentation,
    query_research_corpus,
    query_research_metadata,
    query_postgres_metadata,
    search_web,
    execute_python_code,
    generate_pytorch_ablation_script,
]

literature_llm = llm.bind_tools(literature_tools)
mathematician_llm = llm.bind_tools(mathematician_tools)
experimentalist_llm = llm.bind_tools(experimentalist_tools)


def orchestrator_node(state: ResearchState) -> ResearchState:
    route = _route_from_messages(state.get("messages", []))
    plan = _build_research_plan(route, state)
    return {
        "next_node": route,
        "active_worker": route,
        "research_plan": plan,
    }


def literature_reviewer_node(state: ResearchState) -> ResearchState:
    response = literature_llm.invoke(
        [
            SystemMessage(content=_worker_prompt(LITERATURE_REVIEWER_PROMPT, state)),
            *state.get("messages", []),
        ]
    )
    updates: ResearchState = {"messages": [response], "active_worker": "literature_reviewer"}
    if response.content and not getattr(response, "tool_calls", None):
        updates["literature_notes"] = [_format_worker_note("Literature Reviewer", str(response.content))]
        updates["draft_report"] = str(response.content)
    return updates


def mathematician_node(state: ResearchState) -> ResearchState:
    response = mathematician_llm.invoke(
        [
            SystemMessage(content=_worker_prompt(MATHEMATICIAN_PROMPT, state)),
            *state.get("messages", []),
        ]
    )
    updates: ResearchState = {"messages": [response], "active_worker": "mathematician"}
    if response.content and not getattr(response, "tool_calls", None):
        updates["math_scratchpad"] = [_format_worker_note("Mathematician", str(response.content))]
        updates["draft_report"] = str(response.content)
    return updates


def experimentalist_node(state: ResearchState) -> ResearchState:
    response = experimentalist_llm.invoke(
        [
            SystemMessage(content=_worker_prompt(EXPERIMENTALIST_PROMPT, state)),
            *state.get("messages", []),
        ]
    )
    updates: ResearchState = {"messages": [response], "active_worker": "experimentalist"}
    if response.content and not getattr(response, "tool_calls", None):
        updates["code_scratchpad"] = [_format_worker_note("Experimentalist", str(response.content))]
        updates["draft_report"] = str(response.content)
    return updates


def route_from_orchestrator(state: ResearchState) -> RouteName:
    return state.get("next_node", "literature_reviewer")


def route_back_to_worker(state: ResearchState) -> RouteName:
    return state.get("active_worker", "literature_reviewer")


def _route_from_messages(messages: list[BaseMessage]) -> RouteName:
    latest = _latest_human_text(messages).lower()
    math_terms = {
        "derive",
        "derivation",
        "proof",
        "prove",
        "symbolic",
        "sympy",
        "gradient",
        "hessian",
        "eigen",
        "matrix",
        "equation",
        "loss",
        "optimize",
        "optimization",
    }
    experiment_terms = {
        "ablation",
        "pytorch",
        "torch",
        "experiment",
        "training script",
        "baseline",
        "train loop",
        "benchmark",
    }
    if any(term in latest for term in experiment_terms):
        return "experimentalist"
    if any(term in latest for term in math_terms):
        return "mathematician"
    return "literature_reviewer"


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            return str(message.content)
    return str(messages[-1].content) if messages else ""


def _build_research_plan(route: WorkerName, state: ResearchState) -> str:
    query = _latest_human_text(state.get("messages", []))
    if route == "mathematician":
        return f"1. Formalize the requested derivation/check.\n2. Use sympy/numpy if needed.\n3. Save proof notes to math_scratchpad.\nQuery: {query}"
    if route == "experimentalist":
        return f"1. Identify the experiment or ablation target.\n2. Use local paper context if needed.\n3. Save runnable code to code_scratchpad.\nQuery: {query}"
    return f"1. Search the local research corpus first.\n2. Check metadata/source identity when useful.\n3. Save findings to literature_notes.\nQuery: {query}"


def _worker_prompt(base_prompt: str, state: ResearchState) -> str:
    return (
        f"{base_prompt}\n\n"
        "Shared Memory Scratchpad:\n"
        f"Research Plan:\n{state.get('research_plan', '')}\n\n"
        f"Literature Notes:\n{_join_notes(state.get('literature_notes', []))}\n\n"
        f"Math Scratchpad:\n{_join_notes(state.get('math_scratchpad', []))}\n\n"
        f"Code Scratchpad:\n{_join_notes(state.get('code_scratchpad', []))}\n\n"
        f"Draft Report:\n{state.get('draft_report', '')}\n\n"
        "Write durable findings into your assigned scratchpad through your final response. "
        "Use the main chat only for tool protocol and the final user-facing answer."
    )


def _join_notes(notes: list[str]) -> str:
    return "\n\n".join(notes[-8:]) if notes else "<empty>"


def _format_worker_note(worker: str, content: str) -> str:
    return f"[{worker}]\n{content}"


workflow = StateGraph(ResearchState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("literature_reviewer", literature_reviewer_node)
workflow.add_node("mathematician", mathematician_node)
workflow.add_node("experimentalist", experimentalist_node)
workflow.add_node("tools", ToolNode(tools=all_tools))

workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    {
        "literature_reviewer": "literature_reviewer",
        "mathematician": "mathematician",
        "experimentalist": "experimentalist",
    },
)
workflow.add_conditional_edges("literature_reviewer", tools_condition)
workflow.add_conditional_edges("mathematician", tools_condition)
workflow.add_conditional_edges("experimentalist", tools_condition)
workflow.add_conditional_edges(
    "tools",
    route_back_to_worker,
    {
        "literature_reviewer": "literature_reviewer",
        "mathematician": "mathematician",
        "experimentalist": "experimentalist",
    },
)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    initial_state = {"messages": [HumanMessage(content="What is Chinchilla training?")]}
    for event in app.stream(initial_state, {"configurable": {"thread_id": "agent-smoke"}}):
        for node_name, node_state in event.items():
            print(f"Update from node '{node_name}':")
            if "messages" in node_state:
                print(node_state["messages"][-1].content)
