from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import server
from src.rag.settings import RagSettings


def test_health_endpoint():
    client = TestClient(server.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_web_index_is_served():
    client = TestClient(server.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "OmniRouter Research Scientist" in response.text


def test_chat_stream_returns_cached_tokens(monkeypatch):
    monkeypatch.setattr(server, "check_cache", lambda query: "cached answer")
    saved = []
    monkeypatch.setattr(server, "save_to_cache", lambda query, answer: saved.append((query, answer)))
    client = TestClient(server.app)

    response = client.post("/chat/stream", json={"query": "hello", "use_cache": True})

    assert response.status_code == 200
    assert "event: meta" in response.text
    assert '"source": "cache"' in response.text
    assert "cached " in response.text
    assert saved == []


def test_chat_stream_uses_agent_when_cache_disabled(monkeypatch):
    async def fake_events(initial_state, config, version):
        yield {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="hello ")}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="world")}}

    monkeypatch.setattr(server, "check_cache", lambda query: None)
    saved = []
    monkeypatch.setattr(server, "save_to_cache", lambda query, answer: saved.append((query, answer)))
    monkeypatch.setattr(server.agent_app, "astream_events", fake_events)
    client = TestClient(server.app)

    response = client.post(
        "/chat/stream",
        json={"query": "hello", "thread_id": "test-thread", "use_cache": False},
    )

    assert response.status_code == 200
    assert '"thread_id": "test-thread"' in response.text
    assert "hello " in response.text
    assert "world" in response.text
    assert saved == []


def test_upload_documents_saves_to_data_lake_and_syncs(monkeypatch, tmp_path):
    settings = RagSettings(project_root=tmp_path)
    sync_calls = []

    def fake_sync_corpus(settings, force):
        sync_calls.append((settings.data_lake_dir, force))
        return SimpleNamespace(
            rebuilt=True,
            version_label="version_v1",
            source_count=1,
            chunk_count=1,
        )

    monkeypatch.setattr(server, "get_rag_settings", lambda: settings)
    monkeypatch.setattr(server, "sync_corpus", fake_sync_corpus)
    client = TestClient(server.app)

    response = client.post(
        "/documents/upload",
        files=[("files", ("paper.txt", b"research note", "text/plain"))],
        data={"sync": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["uploaded"][0]["path"] == "data_lake/paper.txt"
    assert payload["rebuilt"] is True
    assert payload["chunk_count"] == 1
    assert (tmp_path / "data_lake" / "paper.txt").read_text(encoding="utf-8") == "research note"
    assert sync_calls == [(settings.data_lake_dir, True)]


def test_upload_documents_rejects_unsupported_file(monkeypatch, tmp_path):
    settings = RagSettings(project_root=tmp_path)
    monkeypatch.setattr(server, "get_rag_settings", lambda: settings)
    client = TestClient(server.app)

    response = client.post(
        "/documents/upload",
        files=[("files", ("script.exe", b"nope", "application/octet-stream"))],
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.text


def test_list_documents_reads_data_lake(monkeypatch, tmp_path):
    settings = RagSettings(project_root=tmp_path)
    settings.data_lake_dir.mkdir(parents=True)
    (settings.data_lake_dir / "paper.md").write_text("# Paper", encoding="utf-8")
    (settings.data_lake_dir / "README.md").write_text("ignore", encoding="utf-8")
    monkeypatch.setattr(server, "get_rag_settings", lambda: settings)
    client = TestClient(server.app)

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["documents"] == [
        {"filename": "paper.md", "path": "data_lake/paper.md", "size_bytes": 7}
    ]
