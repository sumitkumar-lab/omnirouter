from src.evaluation.cli import load_execution_records, run_cli


def test_load_execution_records_builds_query_records_from_json_fixtures():
    records = load_execution_records(
        "tests/fixtures/evaluation_dataset.json",
        "tests/fixtures/evaluation_records.json",
    )

    assert len(records) == 2
    assert records[0].sample.query == "What is OmniRouter?"
    assert records[0].retrieved_docs[0].doc_id == "doc-2"
    assert records[0].reranked_docs[0].doc_id == "doc-1"


def test_run_cli_executes_full_pipeline_and_writes_report(tmp_path):
    report = run_cli(
        dataset_path="tests/fixtures/evaluation_dataset.json",
        records_path="tests/fixtures/evaluation_records.json",
        output_path=tmp_path / "cli_report.json",
        retrieval_k=2,
    )

    assert report.summary.total_queries == 2
    assert report.report_path == tmp_path / "cli_report.json"
    assert report.report_path.exists()
