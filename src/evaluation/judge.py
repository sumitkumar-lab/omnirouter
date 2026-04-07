"""
We are going to create a strict grading script using LangChain's 
**with_structured_output**. This forces our Judge LLM to return a strict JSON 
object containing an integer score (1 for Pass, 0 for Fail) and a reasoning string.
"""
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 1. The Strict Grading Schema
# ==========================================
class HallucinationScore(BaseModel):
    score: int = Field(description="Return 1 if perfectly grounded. Return 0 if hallucinated.")
    reasoning: str = Field(description="A 1-sentence explanation of why you gave this score.")

# ==========================================
# 2. Initialize the Impartial Judge
# ==========================================
# We use temperature=0 because we want strict, deterministic grading, not creativity!
model_name_1 = "llama-3.1-70b-versatile"
model_name_2 = "llama-3.1-8b-instant"

judge_llm = ChatGroq(model=model_name_2, temperature=0)
structured_judge = judge_llm.with_structured_output(HallucinationScore)

# ==========================================
# 3. The Grading Rubric (System Prompt)
# ==========================================
system_prompt = """You are an impartial AI Compliance Judge evaluating an Agent's response.
You will be given the 'Retrieved Context' from the database, and the 'Agent Answer'.
Your ONLY job is to check for HALLUCINATIONS.

RULE:
- If the Agent's answer contains ANY factual information, names, or numbers that are NOT present in the Retrieved Context, score it a 0.
- If the Agent's answer is strictly based ONLY on the context, score it a 1.
- Do not grade grammar or tone. Only grade factual grounding.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Retrieved Context: \n\n {context} \n\n Agent Answer: \n\n {answer}")
])

evaluator = prompt | structured_judge

def check_hallucination(context: str, answer: str):
    print("\n⚖️ [JUDGE] Evaluating answer for hallucinations...")
    try:
        result = evaluator.invoke({"context": context, "answer": answer})
        return result
    except Exception as e:
        print(f"Judge Error: {e}")
        return None
    

if __name__ == "__main__":
    # The reality: What our Vector DB actually found.
    simulated_context = (
        "OmniRouter is an AI architecture that routes LLM requests. "
        "It supports OpenAI and Anthropic APIs."
    )
    
    print("\n========== TEST 1: The Good Agent ==========")
    good_answer = "OmniRouter routes requests and works with Anthropic and OpenAI."
    good_result = check_hallucination(simulated_context, good_answer)
    print(f"Score: {good_result.score}/1")
    print(f"Reasoning: {good_result.reasoning}")

    print("\n========== TEST 2: The Hallucinating Agent ==========")
    bad_answer = "OmniRouter routes requests and works with OpenAI, Anthropic, and Google Gemini."
    bad_result = check_hallucination(simulated_context, bad_answer)
    print(f"Score: {bad_result.score}/1")
    print(f"Reasoning: {bad_result.reasoning}")