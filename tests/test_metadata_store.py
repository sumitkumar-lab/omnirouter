from src.rag.metadata_store import MetadataStore, SourceDocument


def test_metadata_store_records_versions_and_indices(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'metadata.db').as_posix()}"
    store = MetadataStore(db_url)
    store.init_db()

    version = store.create_corpus_version(
        version_label="version_v1",
        manifest={
            "data_lake/source_name/doc.txt": {
                "sha256": "abc123",
                "size": 10,
                "mtime_ns": 123456789,
            }
        },
        source_count=1,
        chunk_count=2,
        index_path=tmp_path / "corpus" / "version_v1" / "faiss_index",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    latest_version = store.get_latest_corpus_version()
    latest_index = store.get_latest_index()

    assert version.version_label == "version_v1"
    assert latest_version is not None and latest_version.source_count == 1
    assert latest_index is not None and latest_index.index_path.endswith("faiss_index")


def test_metadata_store_uses_file_name_for_flat_data_lake_sources(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'metadata.db').as_posix()}"
    store = MetadataStore(db_url)
    store.init_db()

    store.create_corpus_version(
        version_label="version_v1",
        manifest={
            "data_lake/omnirouter_facts.txt": {
                "sha256": "abc123",
                "size": 10,
                "mtime_ns": 123456789,
            }
        },
        source_count=1,
        chunk_count=1,
        index_path=None,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    with store.session_factory() as session:
        source = session.query(SourceDocument).one()

    assert source.source_name == "omnirouter_facts"
