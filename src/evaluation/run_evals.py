from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# Import your real compiled agent and your real judge
from src.agent.graph import app
from src.evaluation.judge import check_hallucination

def evaluate_real_agent(query: str):
    print(f"\n==================================================")
    print(f"🚀 RUNNING REAL AGENT EVALUATION")
    print(f"Query: '{query}'")
    print(f"==================================================")
    
    # 1. Trigger the Real Agent
    initial_state = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": "automated_eval_run_1"}}
    
    print("\n🤖 Agent is thinking and searching...")
    # We use .invoke() here because we don't need streaming for a backend test
    final_state = app.invoke(initial_state, config) 
    
    # 2. Extract the Dynamic Data from the State Machine's Memory
    retrieved_context = ""
    final_answer = ""
    
    for msg in final_state["messages"]:
        # Find the exact text the ChromaDB tool returned
        if isinstance(msg, ToolMessage):
            retrieved_context += msg.content + "\n"
        # Find the final answer the Agent generated
        elif isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            
    if not retrieved_context:
        print("⚠️ Agent did not use the database. Cannot run Hallucination check.")
        return

    # 3. Pass the dynamic data to the Judge
    result = check_hallucination(context=retrieved_context, answer=final_answer)
    
    # 4. Print the final Evaluation Report
    print(f"\n📊 EVALUATION REPORT")
    print(f"Score: {result.score} / 1")
    if result.score == 1:
        print("✅ PASS: Answer is completely grounded in the database.")
    else:
        print("❌ FAIL: Hallucination detected!")
        
    print(f"Judge's Reasoning: {result.reasoning}")
    print(f"==================================================\n")

if __name__ == "__main__":
    # Test our agent with a real query!
    evaluate_real_agent("What is OmniRouter and what does it do?")