"""
The most important part of a robust state machine is strict typing. 
We use Python's TypedDict to define exactly what our shared memory looks like.
"""
# ==========================
# State -> The Central Truth
# ==========================
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

# 1. Define the State
class AgentState(TypedDict):
    """
    This is the shared memory dictionary. 
    Every node will receive this, and every node must return a piece of this to update it.
    """
    messages: List[BaseMessage]  # The chat history
    current_intent: str          # What the agent thinks the user wants
    local_search_failed: bool    # A flag to trigger our fallback loop


# ===================
# Nodes -> The Worker
# ===================
def intent_router_node(state: AgentState):
    """
    Node 1: Analyzes the user's message and decides what to do.
    """
    print("--- [NODE: Intent Router] Analyzing message... ---")
    last_message = state["messages"][-1].content
    
    # Mocking an LLM decision
    if "search" in last_message.lower() or "find" in last_message.lower():     # last_message.lower() instead of last_message
        intent = "needs_search"
    else:
        intent = "direct_answer"
        
    # We return ONLY the parts of the state we want to update
    return {"current_intent": intent}

def local_db_search_node(state: AgentState):
    """
    Node 2: Attempts to search our ChromaDB.
    """
    print("--- [NODE: Local DB Search] Searching internal docs... ---")
    
    # Mocking a database failure
    # Let's pretend the database didn't find anything relevant
    print("    -> Result: No relevant chunks found.")
    
    return {"local_search_failed": True}

def direct_answer_node(state: AgentState):
    """
    Node 3: Generates the final response.
    """
    print("--- [NODE: Direct Answer] Generating response... ---")
    
    if state.get("local_search_failed"):
        response = "I couldn't find that in my local database. I need to search the web."
    else:
        response = "I am ready to help you with AI Engineering."
        
    return {"messages": state["messages"] + [{"role": "assistant", "content": response}]}


# =================
# Edge -> The Brain
# =================
from langgraph.graph import StateGraph, END

# The Conditional Edge Logic
def route_after_intent(state: AgentState):
    """
    Looks at the state and returns the name of the NEXT node to go to.
    """
    if state["current_intent"] == "needs_search":
        return "local_search"
    else:
        return "direct_answer"

# --- COMPILE THE GRAPH ---
workflow = StateGraph(AgentState)

# 1. Add our nodes to the graph
workflow.add_node("intent_router", intent_router_node)
workflow.add_node("local_search", local_db_search_node)
workflow.add_node("direct_answer", direct_answer_node)

# 2. Define the Entry Point
workflow.set_entry_point("intent_router")

# 3. Add Conditional Edges
# "From intent_router, run this function to decide where to go next."
workflow.add_conditional_edges(
    "intent_router",
    route_after_intent,
    {
        # Map the function's return string to the actual node name
        "local_search": "local_search",
        "direct_answer": "direct_answer"
    }
)

# 4. Add Standard Edges
# "After local_search finishes, ALWAYS go to direct_answer."
workflow.add_edge("local_search", "direct_answer")
workflow.add_edge("direct_answer", END)

# Compile into an executable application
app = workflow.compile()


# ====
# Test
# ====
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    # Test 1: A direct chat message
    print("\n========== TEST 1 ==========")
    initial_state_1 = {
        "messages": [HumanMessage(content="Hello, how are you?")],
        "current_intent": "",
        "local_search_failed": False
    }
    app.invoke(initial_state_1)

    # Test 2: A search request
    print("\n========== TEST 2 ==========")
    initial_state_2 = {
        "messages": [HumanMessage(content="Can you search the documentation for LangGraph?")],
        "current_intent": "",
        "local_search_failed": False
    }
    app.invoke(initial_state_2)