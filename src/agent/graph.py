import os
from typing import Annotated, TypedDict
# # OpenAI...
# from langchain_openai import ChatOpenAI
# Groq LLM...
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from src.agent.tools import search_documentation
# Import our custom tool
from src.agent.tools import search_documentation
# Fix the infinite loop using system prompt(Guardrail)
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
load_dotenv()


# 1. UPGRADED STATE
class AgentState(TypedDict):
    # 'add_messages' ensures we append to the history, never overwrite it.
    messages: Annotated[list[BaseMessage], add_messages]

# 2. INITIALIZE THE BRAIN
# We instantiate the LLM and "bind" our tool to it. 

# # Make sure you export GROQ_API_KEY in your terminal before running!
# os.environ["GROQ_API_KEY"] = "gsk_jd"
# We use Meta's Llama 3 8B model hosted on Groq for incredible speed
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# # This sends the JSON schema we looked at yesterday directly to OpenAI.
# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

tools = [search_documentation]
llm_with_tools = llm.bind_tools(tools)

# 3. THE NODES
"""
Chat node for seamless interaction with the LLM model...
"""
def chatbot_node(state: AgentState):
    """
    This node intercepts the history, adds strict behavioral rules, 
    and then passes it to the LLM.

    The LLM will either return a standard text message, OR a special "ToolCall" message.
    """
    print("\n--- [NODE: Chatbot] Thinking... ---")

    # 1. The Circuit Breaker System Prompt --> Prompt based Flow control
    system_message = SystemMessage(content=(
        "You are an elite AI Engineering assistant. "
        "You have access to a search_documentation tool. "
        "CRITICAL RULE: If you use the tool and it returns 'No relevant information found', "
        "you MUST NOT use the tool again. Immediately stop and tell the user "
        "'I do not have enough information to answer that based on the documentation.' "
        "Do not guess. Do not hallucinate."
    ))
    
    # 2. Prepend the system rules to the chat history
    messages_to_send = [system_message] + state["messages"]

    # 3. Invoke the LLM with the strict rules applied
    response = llm_with_tools.invoke(messages_to_send)
    
    # We return the message wrapped in a list to trigger the 'add_messages' append behavior
    return {"messages": [response]}

# LangGraph has a built-in node specifically for executing tools!
# It reads the "ToolCall" message, runs our Python function, and returns a "ToolMessage".
tool_node = ToolNode(tools=tools)

# 4. COMPILE THE GRAPH
workflow = StateGraph(AgentState)

# Add our two worker nodes
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.add_edge(START, "chatbot")

# 5. THE MAGIC ROUTING
# 'tools_condition' is a built-in LangGraph edge.
# It looks at the last message from the chatbot. 
# If it has a tool call, it routes to "tools". If it's just text, it routes to END.
workflow.add_conditional_edges(
    "chatbot",
    tools_condition, 
)

# After a tool finishes running, ALWAYS loop back to the chatbot 
# so it can read the database results and formulate a final answer!
workflow.add_edge("tools", "chatbot")

app = workflow.compile()


from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    # Ensure your API key is available
    os.environ["OPENAI_API_KEY"] = "YOUR_REAL_OPENAI_API_KEY"
    
    print("========== AGENT TEST ==========")
    initial_state = {
        "messages": [HumanMessage(content="What does OmniRouter do?")]
    }
    
    # stream() allows us to see the exact output of each node as it executes!
    for event in app.stream(initial_state):
        for node_name, node_state in event.items():
            print(f"Update from node '{node_name}':")
            # Print the content of the very last message added to the state
            print(f" -> {node_state['messages'][-1].content}\n")