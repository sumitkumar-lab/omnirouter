from src.rag.pipeline import sync_corpus
from src.rag.settings import RagSettings


def test_sync_corpus_versions_and_rebuilds_on_change(tmp_path):
    raw_dir = tmp_path / "data_lake" / "support"
    raw_dir.mkdir(parents=True)
    source_file = raw_dir / "runbook.txt"
    source_file.write_text("Version one runbook.", encoding="utf-8")

    settings = RagSettings(project_root=tmp_path)

    first = sync_corpus(settings=settings, force=False)
    second = sync_corpus(settings=settings, force=False)

    source_file.write_text("Version two runbook.", encoding="utf-8")
    third = sync_corpus(settings=settings, force=False)

    assert first.rebuilt is True
    assert first.version_label == "version_v1"
    assert second.rebuilt is False
    assert third.rebuilt is True
    assert third.version_label == "version_v2"
    assert (tmp_path / "corpus" / "version_v1" / "manifest.json").exists()
    assert (tmp_path / "corpus" / "version_v2" / "chunks.jsonl").exists()
