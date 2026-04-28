from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.schemas import EvaluationSample


def load_evaluation_dataset(dataset_path: str | Path) -> list[EvaluationSample]:
    path = Path(dataset_path)
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("Evaluation dataset must be a JSON array of samples.")
    return [EvaluationSample.model_validate(item) for item in raw_data]
