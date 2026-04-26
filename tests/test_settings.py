from src.rag.settings import RagSettings


def test_settings_default_storage_layout(tmp_path):
    settings = RagSettings(project_root=tmp_path)

    assert settings.data_lake_dir == tmp_path / "data_lake"
    assert settings.corpus_dir == tmp_path / "corpus"
    assert settings.metadata_db_url.startswith("sqlite:///")
