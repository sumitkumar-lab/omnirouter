from types import SimpleNamespace

from src.evaluation.judge import HallucinationScore, check_hallucination


class FakeEvaluator:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.result


class FailingEvaluator:
    def invoke(self, payload):
        raise RuntimeError("judge offline")


def test_check_hallucination_uses_evaluator_chain():
    fake_result = HallucinationScore(score=1, reasoning="Grounded.")
    fake_evaluator = FakeEvaluator(fake_result)

    result = check_hallucination(
        context="OmniRouter supports OpenAI.",
        answer="OmniRouter supports OpenAI.",
        evaluator_chain=fake_evaluator,
    )

    assert result == fake_result
    assert fake_evaluator.calls == [
        {
            "context": "OmniRouter supports OpenAI.",
            "answer": "OmniRouter supports OpenAI.",
        }
    ]


def test_check_hallucination_returns_none_on_evaluator_error():
    result = check_hallucination(
        context="ctx",
        answer="ans",
        evaluator_chain=FailingEvaluator(),
    )

    assert result is None
