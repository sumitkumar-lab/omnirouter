from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.schemas import EvaluationReport


def save_evaluation_report(report: EvaluationReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def print_evaluation_summary(report: EvaluationReport) -> None:
    summary = report.summary
    print("Evaluation Summary")
    print(f"avg Recall@k: {summary.avg_recall_at_k:.4f}")
    print(f"avg Precision@k: {summary.avg_precision_at_k:.4f}")
    print(f"avg MRR: {summary.avg_mrr:.4f}")
    print(f"faithfulness score: {summary.avg_faithfulness_score:.4f}")
    print(f"hallucination rate: {summary.hallucination_rate:.4f}")
