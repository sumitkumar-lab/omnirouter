from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.dataset import load_evaluation_dataset
from src.evaluation.orchestrator import EvaluationOrchestrator
from src.evaluation.schemas import (
    AgentTraceStep,
    EvaluationSample,
    QueryExecutionRecord,
    RetrievedDocument,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation pipeline from JSON inputs.")
    parser.add_argument("--dataset", required=True, help="Path to the evaluation dataset JSON file.")
    parser.add_argument(
        "--records",
        required=True,
        help="Path to the execution-records JSON file with retrieval, rerank, answer, and trace data.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Path to save the structured evaluation report JSON.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional retrieval cutoff for Recall@k and Precision@k.",
    )
    return parser


def load_execution_records(
    dataset_path: str | Path,
    records_path: str | Path,
) -> list[QueryExecutionRecord]:
    samples = load_evaluation_dataset(dataset_path)
    sample_map = {sample.query: sample for sample in samples}
    payload = json.loads(Path(records_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Execution records file must be a JSON array.")

    records: list[QueryExecutionRecord] = []
    for item in payload:
        sample = _resolve_sample(item, sample_map)
        retrieved_docs = [RetrievedDocument.model_validate(doc) for doc in item.get("retrieved_docs", [])]
        reranked_docs = [RetrievedDocument.model_validate(doc) for doc in item.get("reranked_docs", [])]
        trace = [AgentTraceStep.model_validate(step) for step in item.get("trace", [])]
        records.append(
            QueryExecutionRecord(
                sample=sample,
                retrieved_docs=retrieved_docs,
                reranked_docs=reranked_docs or retrieved_docs,
                final_answer=item.get("final_answer", ""),
                retrieved_context=item.get("retrieved_context", ""),
                trace=trace,
            )
        )
    return records


def run_cli(dataset_path: str | Path, records_path: str | Path, output_path: str | Path, retrieval_k: int | None):
    records = load_execution_records(dataset_path, records_path)
    orchestrator = EvaluationOrchestrator()
    return orchestrator.evaluate_records(records, retrieval_k=retrieval_k, output_path=output_path)


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    report = run_cli(
        dataset_path=args.dataset,
        records_path=args.records,
        output_path=args.output,
        retrieval_k=args.k,
    )
    print(f"Saved report to {report.report_path}")
    return 0


def _resolve_sample(item: dict, sample_map: dict[str, EvaluationSample]) -> EvaluationSample:
    query = item.get("query")
    if not query:
        raise ValueError("Each execution record must include a query.")
    try:
        return sample_map[query]
    except KeyError as exc:
        raise ValueError(f"Execution record query not found in dataset: {query}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
