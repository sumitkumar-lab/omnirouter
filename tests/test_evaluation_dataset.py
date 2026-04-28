from src.evaluation.dataset import load_evaluation_dataset


def test_load_evaluation_dataset_reads_samples():
    samples = load_evaluation_dataset("tests/fixtures/evaluation_dataset.json")

    assert len(samples) == 2
    assert samples[0].query == "What is OmniRouter?"
    assert samples[0].relevant_doc_ids == ["doc-1", "doc-3"]
