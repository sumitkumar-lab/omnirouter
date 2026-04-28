from typing import Annotated, Literal, TypedDict
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

WorkerName = Literal["literature_reviewer", "mathematician", "experimentalist"]


class ResearchState(TypedDict, total=False):
    # Main user-facing chat history. Keep this small: user prompts, final answers,
    # and tool-call protocol messages required by LangGraph.
    messages: Annotated[list[BaseMessage], add_messages]

    # The orchestrator uses this to tell LangGraph who to wake up next.
    next_node: WorkerName
    active_worker: WorkerName

    # Shared memory scratchpad. Workers pass durable research artifacts here
    # instead of stuffing every intermediate note into the chat history.
    research_plan: str
    literature_notes: Annotated[list[str], operator.add]
    math_scratchpad: Annotated[list[str], operator.add]
    code_scratchpad: Annotated[list[str], operator.add]

    # Draft answer/report being assembled across workers.
    draft_report: str
