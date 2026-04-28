from __future__ import annotations

import ast
import contextlib
import io
import os
import sqlite3
import textwrap
from typing import Any

import numpy as np
import requests
import sympy as sp
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from pydantic import BaseModel, Field

from src.rag.metadata_store import MetadataStore
from src.rag.settings import get_rag_settings
from src.rag.vector_store import get_vector_store


class DocumentGrader(BaseModel):
    is_relevant: str = Field(description="Return 'yes' if the document is relevant to the query, 'no' if not.")


grader_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_grader = grader_llm.with_structured_output(DocumentGrader)
grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Score whether the retrieved research excerpt is useful for the query. "
            "Return 'yes' only when it contains helpful definitions, equations, claims, methods, or citations.",
        ),
        ("human", "Query: {query}\n\nDocument: {document}"),
    ]
)
grader_chain = grader_prompt | structured_grader


@tool
def search_documentation(query: str) -> str:
    """
    Search the single local research corpus built from files in data_lake.

    Use this for literature review, paper-specific questions, equations,
    architecture details, proof sketches, and uploaded research notes.
    """
    try:
        db = get_vector_store()
        base_retriever = db.as_retriever(search_kwargs={"k": 10})
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
        results = compression_retriever.invoke(query)
        if not results:
            return "No documents found in the local research corpus."

        valid_docs: list[str] = []
        for doc in results:
            score = grader_chain.invoke({"query": query, "document": doc.page_content})
            if score.is_relevant.lower() == "yes":
                metadata = getattr(doc, "metadata", None) or {}
                source = metadata.get("source")
                header = metadata.get("chunk_header")
                if source:
                    label = f"Source: {source}" + (f"\nSection: {header}" if header else "")
                    valid_docs.append(f"{label}\n{doc.page_content}")
                else:
                    valid_docs.append(doc.page_content)

        if not valid_docs:
            return (
                "INTERNAL SYSTEM DIRECTIVE: The local database does not contain the answer to this query. "
                "The documents retrieved were irrelevant. You MUST immediately use the 'search_web' tool."
            )
        return "\n\n---\n\n".join(valid_docs)
    except Exception as exc:
        return f"Error executing local research search: {exc}"


@tool
def query_research_corpus(query: str) -> str:
    """
    Search the uploaded research papers in data_lake using the canonical FAISS retriever.
    """
    return search_documentation.invoke({"query": query})


@tool
def query_research_metadata(query: str) -> str:
    """
    Search local metadata for ingested data_lake sources and FAISS index artifacts.

    This is the metadata/citation-graph hook for Phase 2. The current project
    uses the configured metadata store; when POSTGRES is configured through the
    RAG metadata URL, the same read path remains the public tool contract.
    """
    settings = get_rag_settings()
    store = MetadataStore(settings.metadata_db_url)
    store.init_db()
    latest = store.get_latest_corpus_version()
    latest_index = store.get_latest_index()
    if latest is None:
        return "No corpus metadata is available yet."

    lines = [
        f"Corpus version: {latest.version_label}",
        f"Sources: {latest.source_count}",
        f"Chunks: {latest.chunk_count}",
    ]
    if latest_index is not None:
        lines.append(f"FAISS index: {latest_index.index_path}")
        lines.append(f"Embedding model: {latest_index.embedding_model}")
    matching_sources = [
        path
        for path in latest.manifest
        if query.lower() in path.lower() or any(token in path.lower() for token in query.lower().split())
    ]
    if matching_sources:
        lines.append("Matching sources:")
        lines.extend(f"- {source}" for source in matching_sources[:10])
    return "\n".join(lines)


@tool
def query_postgres_metadata(sql: str) -> str:
    """
    Execute a read-only PostgreSQL metadata/citation query.

    Set RESEARCH_POSTGRES_DSN or POSTGRES_DSN to enable this tool. Only SELECT
    statements are accepted.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: For security reasons, only SELECT queries are allowed."
    dsn = os.getenv("RESEARCH_POSTGRES_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        return "PostgreSQL metadata is not configured. Set RESEARCH_POSTGRES_DSN or POSTGRES_DSN."
    try:
        import psycopg

        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchmany(25)
                if not rows:
                    return "Query executed successfully, but returned 0 rows."
                return "PostgreSQL Results:\n" + "\n".join(str(row) for row in rows)
    except Exception as exc:
        return f"PostgreSQL query error: {exc}"


@tool
def search_web(query: str) -> str:
    """
    Search DuckDuckGo for external literature, current papers, or missing context.
    """
    try:
        return DuckDuckGoSearchRun().invoke(query)
    except Exception as exc:
        return f"Error executing web search: {exc}"


DISALLOWED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "requests",
    "urllib",
    "http",
    "builtins",
}
DISALLOWED_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint"}


class _SafePythonVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".", 1)[0] in DISALLOWED_IMPORTS:
                raise ValueError(f"Import not allowed: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module.split(".", 1)[0] in DISALLOWED_IMPORTS:
            raise ValueError(f"Import not allowed: {module}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALLS:
            raise ValueError(f"Call not allowed: {node.func.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed.")
        self.generic_visit(node)


@tool
def execute_python_code(code: str) -> str:
    """
    Execute bounded derivation-checking Python with sympy and numpy preloaded.

    Use this for algebra, numeric checks, matrix identities, optimization
    derivations, and small deterministic simulations. The environment blocks
    filesystem, network, subprocess, and dynamic-code primitives.
    """
    try:
        tree = ast.parse(code, mode="exec")
        _SafePythonVisitor().visit(tree)
        output = io.StringIO()
        globals_dict: dict[str, Any] = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "pow": pow,
                "print": print,
                "range": range,
                "round": round,
                "set": set,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
            },
            "np": np,
            "numpy": np,
            "sp": sp,
            "sympy": sp,
        }
        with contextlib.redirect_stdout(output):
            exec(compile(tree, "<secure_math_repl>", "exec"), globals_dict, {})
        result = output.getvalue().strip()
        return f"Terminal Output:\n{result}" if result else "Terminal Output:\n<no printed output>"
    except Exception as exc:
        return f"Error executing Python: {exc}"


@tool
def generate_pytorch_ablation_script(experiment_goal: str) -> str:
    """
    Generate a raw PyTorch ablation script skeleton for a research hypothesis.
    """
    safe_goal = experiment_goal.strip() or "Compare baseline and ablated model variants."
    return textwrap.dedent(
        f"""
        import argparse
        import random
        import torch
        import torch.nn as nn
        import torch.optim as optim


        class TinyAblationModel(nn.Module):
            def __init__(self, width: int, use_ablation: bool):
                super().__init__()
                self.use_ablation = use_ablation
                self.net = nn.Sequential(
                    nn.Linear(128, width),
                    nn.GELU(),
                    nn.Linear(width, 10),
                )

            def forward(self, x):
                logits = self.net(x)
                return logits * 0.5 if self.use_ablation else logits


        def run_trial(seed: int, width: int, use_ablation: bool):
            torch.manual_seed(seed)
            random.seed(seed)
            model = TinyAblationModel(width=width, use_ablation=use_ablation)
            optimizer = optim.AdamW(model.parameters(), lr=3e-4)
            loss_fn = nn.CrossEntropyLoss()
            x = torch.randn(256, 128)
            y = torch.randint(0, 10, (256,))
            for _ in range(20):
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
            return float(loss.detach())


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--width", type=int, default=256)
            parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
            args = parser.parse_args()
            print("Experiment goal: {safe_goal}")
            for use_ablation in (False, True):
                losses = [run_trial(seed, args.width, use_ablation) for seed in args.seeds]
                label = "ablation" if use_ablation else "baseline"
                print(label, sum(losses) / len(losses), losses)


        if __name__ == "__main__":
            main()
        """
    ).strip()


@tool
def get_github_issue(repo: str, issue_number: int) -> str:
    """
    Fetch and summarize a public GitHub issue.
    """
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            return f"Failed to fetch issue. GitHub API status code: {response.status_code}"
        data = response.json()
        title = data.get("title", "No Title")
        body = data.get("body", "No Body")
        return f"ISSUE TITLE: {title}\n\nISSUE DESCRIPTION:\n{body}"[:2000]
    except Exception as exc:
        return f"Error executing GitHub API call: {exc}"


@tool
def execute_sql_query(query: str) -> str:
    """
    Execute a read-only SQL SELECT query against the legacy local SQLite sales database.
    """
    if not query.strip().upper().startswith("SELECT"):
        return "Error: For security reasons, only SELECT queries are allowed."
    try:
        conn = sqlite3.connect("company.db")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "Query executed successfully, but returned 0 rows."
        return "Query Results:\n" + "\n".join(str(row) for row in rows)
    except sqlite3.Error as exc:
        return f"SQL Execution Error: {exc}"
