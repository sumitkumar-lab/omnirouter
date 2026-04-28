"""
Judge adapters for generation and hallucination evaluation.
"""
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.evaluation.generation import GenerationJudge, JudgeResult

load_dotenv()


class HallucinationScore(BaseModel):
    score: int = Field(description="Return 1 if perfectly grounded. Return 0 if hallucinated.")
    reasoning: str = Field(description="A 1-sentence explanation of why you gave this score.")


SYSTEM_PROMPT = """You are an impartial AI Compliance Judge evaluating an Agent's response.
You will be given the 'Retrieved Context' from the database, and the 'Agent Answer'.
Your ONLY job is to check for HALLUCINATIONS.

RULE:
- If the Agent's answer contains ANY factual information, names, or numbers that are NOT present in the Retrieved Context, score it a 0.
- If the Agent's answer is strictly based ONLY on the context, score it a 1.
- Do not grade grammar or tone. Only grade factual grounding.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Retrieved Context: \n\n {context} \n\n Agent Answer: \n\n {answer}"),
    ]
)


@lru_cache(maxsize=4)
def build_evaluator(model_name: str = "llama-3.1-8b-instant"):
    judge_llm = ChatGroq(model=model_name, temperature=0)
    structured_judge = judge_llm.with_structured_output(HallucinationScore)
    return prompt | structured_judge


class LLMFaithfulnessJudge(GenerationJudge):
    def __init__(self, evaluator_chain=None):
        self.evaluator_chain = evaluator_chain

    def score(self, answer: str, context: str, ground_truth_answer: str) -> JudgeResult:
        result = check_hallucination(context=context, answer=answer, evaluator_chain=self.evaluator_chain)
        if result is None:
            return JudgeResult(
                faithfulness_score=0.0,
                hallucination_score=1.0,
                reasoning="Judge failed to evaluate answer grounding.",
            )
        faithfulness = float(result.score)
        return JudgeResult(
            faithfulness_score=faithfulness,
            hallucination_score=1.0 - faithfulness,
            reasoning=result.reasoning,
        )


def check_hallucination(context: str, answer: str, evaluator_chain=None):
    print("\n[JUDGE] Evaluating answer for hallucinations...")
    try:
        if evaluator_chain is None:
            evaluator_chain = build_evaluator()
        return evaluator_chain.invoke({"context": context, "answer": answer})
    except Exception as exc:
        print(f"Judge Error: {exc}")
        return None
