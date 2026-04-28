from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.evaluation.schemas import GenerationMetrics


@dataclass(slots=True)
class JudgeResult:
    faithfulness_score: float
    hallucination_score: float
    reasoning: str


class GenerationJudge:
    def score(self, answer: str, context: str, ground_truth_answer: str) -> JudgeResult:
        raise NotImplementedError


class HeuristicGenerationJudge(GenerationJudge):
    def score(self, answer: str, context: str, ground_truth_answer: str) -> JudgeResult:
        answer_tokens = set(_tokenize(answer))
        context_tokens = set(_tokenize(context))

        if not answer_tokens:
            return JudgeResult(0.0, 0.0, "Answer is empty.")

        supported_tokens = answer_tokens & context_tokens
        unsupported_ratio = 1.0 - (len(supported_tokens) / len(answer_tokens))
        faithfulness = 1.0 - unsupported_ratio
        reasoning = "Answer tokens are mostly supported by retrieved context."
        if unsupported_ratio > 0.2:
            reasoning = "Answer contains unsupported information compared with retrieved context."

        return JudgeResult(
            faithfulness_score=round(max(0.0, min(1.0, faithfulness)), 4),
            hallucination_score=round(max(0.0, min(1.0, unsupported_ratio)), 4),
            reasoning=reasoning,
        )


def evaluate_generation(
    answer: str,
    ground_truth_answer: str,
    retrieved_context: str,
    judge: GenerationJudge | None = None,
) -> GenerationMetrics:
    judge = judge or HeuristicGenerationJudge()
    semantic_similarity = _cosine_similarity(answer, ground_truth_answer)
    judge_result = judge.score(answer=answer, context=retrieved_context, ground_truth_answer=ground_truth_answer)

    return GenerationMetrics(
        semantic_similarity=semantic_similarity,
        faithfulness_score=judge_result.faithfulness_score,
        hallucination_score=judge_result.hallucination_score,
        hallucination_detected=judge_result.hallucination_score > 0.2,
        judge_reasoning=judge_result.reasoning,
    )


def _cosine_similarity(left: str, right: str) -> float:
    left_counts = _term_counts(_tokenize(left))
    right_counts = _term_counts(_tokenize(right))
    if not left_counts or not right_counts:
        return 0.0

    dot = sum(left_counts[token] * right_counts.get(token, 0) for token in left_counts)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if not left_norm or not right_norm:
        return 0.0
    return round(dot / (left_norm * right_norm), 4)


def _term_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
