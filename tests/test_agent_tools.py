from types import SimpleNamespace

from src.agent import tools as agent_tools


class FakeVectorStore:
    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return "base-retriever"


class FakeCompressionRetriever:
    def __init__(self, base_compressor, base_retriever):
        self.base_compressor = base_compressor
        self.base_retriever = base_retriever

    def invoke(self, query):
        return [
            SimpleNamespace(page_content="OmniRouter routes requests across providers."),
            SimpleNamespace(page_content="Unrelated deployment note."),
        ]


class FakeGraderChain:
    def __init__(self, scores):
        self.scores = scores

    def invoke(self, payload):
        content = payload["document"]
        return SimpleNamespace(is_relevant=self.scores.get(content, "no"))


def test_search_documentation_returns_only_relevant_reranked_docs(monkeypatch):
    monkeypatch.setattr(agent_tools, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(agent_tools, "HuggingFaceCrossEncoder", lambda model_name: SimpleNamespace(model_name=model_name))
    monkeypatch.setattr(agent_tools, "CrossEncoderReranker", lambda model, top_n: SimpleNamespace(model=model, top_n=top_n))
    monkeypatch.setattr(agent_tools, "ContextualCompressionRetriever", FakeCompressionRetriever)
    monkeypatch.setattr(
        agent_tools,
        "grader_chain",
        FakeGraderChain({"OmniRouter routes requests across providers.": "yes"}),
    )

    result = agent_tools.search_documentation.invoke({"query": "What does OmniRouter do?"})

    assert result == "OmniRouter routes requests across providers."


def test_search_documentation_directs_web_search_when_docs_are_irrelevant(monkeypatch):
    monkeypatch.setattr(agent_tools, "get_vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(agent_tools, "HuggingFaceCrossEncoder", lambda model_name: SimpleNamespace(model_name=model_name))
    monkeypatch.setattr(agent_tools, "CrossEncoderReranker", lambda model, top_n: SimpleNamespace(model=model, top_n=top_n))
    monkeypatch.setattr(agent_tools, "ContextualCompressionRetriever", FakeCompressionRetriever)
    monkeypatch.setattr(agent_tools, "grader_chain", FakeGraderChain({}))

    result = agent_tools.search_documentation.invoke({"query": "latest LangGraph release"})

    assert "local database does not contain the answer" in result
    assert "search_web" in result


def test_execute_sql_query_rejects_non_select_statements():
    result = agent_tools.execute_sql_query.invoke({"query": "DELETE FROM sales"})

    assert result == "Error: For security reasons, only SELECT queries are allowed."
