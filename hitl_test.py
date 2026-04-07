import os
from langchain_core.messages import HumanMessage
from src.agent.graph import app

def run_human_in_the_loop():
    print("\n========== ENTERPRISE HITL DASHBOARD ==========")
    
    # We MUST provide a thread_id now so the checkpointer knows which memory to load
    config = {"configurable": {"thread_id": "admin_session_1"}}
    
    user_input = "Can you search the documentation for LangGraph fallback?"
    print(f"User: {user_input}\n")
    
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # 1. Run the graph until it hits a breakpoint
    print("🤖 Agent is thinking...")
    for event in app.stream(initial_state, config):
        for node_name, node_state in event.items():
            print(f"-> Finished Node: {node_name}")
            
    # 2. Check WHY the graph stopped
    snapshot = app.get_state(config)
    
    # If there is a "next" node pending, it means we hit our breakpoint!
    if snapshot.next:
        print("\n🛑 [BREAKPOINT TRIGGERED] Agent wants to execute a Tool.")
        
        # Look at the tool call the LLM generated
        last_message = snapshot.values["messages"][-1]
        print(f"🔧 Proposed Tool Call: {last_message.tool_calls}")
        
        # 3. The Human Decision
        decision = input("\nDo you approve this action? (y/n): ").strip().lower()
        
        if decision == 'y':
            print("\n✅ Action Approved. Resuming Graph...")
            # Passing "None" as the state tells LangGraph to just continue from where it paused
            for event in app.stream(None, config):
                pass 
            
            # Print the final result
            final_state = app.get_state(config)
            print(f"\n🤖 Final Answer: {final_state.values['messages'][-1].content}")
            
        else:
            print("\n❌ Action Denied. We would handle the rejection logic here.")
    else:
        print("\n✅ Graph finished without needing tools.")

if __name__ == "__main__":
    run_human_in_the_loop()