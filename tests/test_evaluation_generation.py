from src.evaluation.generation import GenerationJudge, JudgeResult, evaluate_generation


class FakeJudge(GenerationJudge):
    def score(self, answer: str, context: str, ground_truth_answer: str) -> JudgeResult:
        return JudgeResult(
            faithfulness_score=0.8,
            hallucination_score=0.2,
            reasoning="Mostly grounded.",
        )


def test_evaluate_generation_combines_similarity_and_judge_scores():
    metrics = evaluate_generation(
        answer="OmniRouter routes LLM requests.",
        ground_truth_answer="OmniRouter routes LLM requests.",
        retrieved_context="OmniRouter routes LLM requests through a central router.",
        judge=FakeJudge(),
    )

    assert metrics.semantic_similarity == 1.0
    assert metrics.faithfulness_score == 0.8
    assert metrics.hallucination_detected is False
