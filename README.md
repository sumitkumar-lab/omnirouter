# OmniRouter

OmniRouter is evolving into a single-agent, multi-worker AI Research Scientist. It ingests research papers from one source folder, builds a local FAISS retrieval index, and runs a LangGraph swarm for literature review, mathematical checking, and PyTorch experiment generation.

## What It Does

- Ingests files from `data_lake/`
- Supports PDFs, Markdown, text, CSV, and JSON
- Prefers Nougat/Marker-style Markdown sidecars for research PDFs
- Chunks Markdown by headers instead of raw character count
- Embeds chunks with HuggingFace embeddings
- Stores vectors in local FAISS indexes under `corpus/`
- Tracks corpus/index metadata in `rag_metadata.db`
- Serves a LangGraph research swarm through FastAPI streaming
- Includes a zero-build browser UI that can be served locally or deployed with GitHub Pages

## Architecture

```text
data_lake/
  paper.pdf
  paper.md              optional OCR sidecar from Nougat/Marker
  notes.txt
        |
        v
src.rag.ingestion       loads docs, prefers OCR Markdown, chunks by headers
        |
        v
src.rag.pipeline        versions corpus snapshots and builds FAISS
        |
        v
src.agent.graph         LangGraph AI Research Scientist swarm
        |
        +--> Orchestrator
        +--> Literature Reviewer
        +--> Mathematician
        +--> Experimentalist
```

## Main Folders

```text
src/
  agent/        LangGraph swarm and tools
  api/          FastAPI streaming endpoint and semantic cache
  evaluation/   agent/RAG evaluation utilities
  rag/          ingestion, corpus sync, FAISS retrieval, metadata store
data_lake/      upload research papers and notes here
corpus/         generated versioned FAISS indexes and chunk snapshots
tests/          focused regression tests
web/            static browser UI for the research agent
.github/        GitHub Pages deployment workflow
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...              # optional, used by chatbot fallback paths
RESEARCH_POSTGRES_DSN=postgresql://user:password@localhost:5432/research  # optional
```

The LangGraph swarm currently uses Groq by default in `src/agent/graph.py`, so `GROQ_API_KEY` is the key you need for the agent.

## Add Research Papers

You can upload files through the browser UI or put them directly in `data_lake/`.

Browser upload:

1. Start the API.
2. Open `http://127.0.0.1:8000`.
3. Use the `Documents` panel.
4. Select `.pdf`, `.md`, `.markdown`, `.txt`, `.csv`, or `.json` files.
5. Keep `Sync after upload` enabled to rebuild the FAISS corpus immediately.

Filesystem upload:

```text
data_lake/
  chinchilla_train.pdf
  chinchilla_train.md
  omnirouter_facts.txt
```

For math-heavy papers, generate Markdown with Nougat or Marker and place it next to the PDF using the same filename:

```text
data_lake/
  my_paper.pdf
  my_paper.md
```

When both exist, ingestion uses `my_paper.md` as the text source but keeps metadata tied to `my_paper.pdf`. This preserves LaTeX blocks and lets the pipeline chunk by Markdown headers such as `Methodology`, `Proofs`, `Experiments`, and `Appendix`.

Upload through the API:

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents/upload `
  -F "files=@data_lake/chinchilla_train.pdf" `
  -F "files=@data_lake/chinchilla_train.md" `
  -F "sync=true"
```

List uploaded documents:

```powershell
curl.exe http://127.0.0.1:8000/documents
```

## Build Or Refresh The FAISS Corpus

The app auto-syncs when retrieval runs, but you can force a rebuild:

```powershell
.\.venv\Scripts\python.exe -c "from src.rag.pipeline import sync_corpus; print(sync_corpus(force=True))"
```

Inspect the generated corpus:

```text
corpus/
  version_v*/
    manifest.json
    chunks.jsonl
    faiss_index/
```

Run the ingestion smoke script:

```powershell
.\.venv\Scripts\python.exe tests\ingestion_main.py
```

## Run The API

Start the FastAPI streaming server:

```powershell
.\.venv\Scripts\uvicorn.exe src.api.server:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Streaming endpoint:

```text
POST /chat/stream
Body: {"query": "What is Chinchilla training?"}
```

Document endpoints:

```text
GET  /documents
POST /documents/upload
```

Run the terminal client in another shell:

```powershell
.\.venv\Scripts\python.exe client.py
```

## Run The Web UI

The web UI is in `web/` and needs no build step.

Local backend plus local UI:

```powershell
.\.venv\Scripts\uvicorn.exe src.api.server:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

GitHub Pages UI:

1. Push the repo to GitHub.
2. In the GitHub repository, open `Settings -> Pages`.
3. Set the source to `GitHub Actions`.
4. Push changes to `main`, or manually run `Deploy web UI to GitHub Pages`.
5. Open the Pages URL and set `API base URL` to your running backend, for example:

```text
http://127.0.0.1:8000
```

For a public GitHub Pages site, the browser can only call an API URL that is reachable from your browser. During local development, keep the FastAPI server running on your machine. For a public deployment, host the FastAPI backend on a service such as Render, Railway, Fly.io, Hugging Face Spaces, or your own server, then paste that backend URL into the web UI.

## Use The LangGraph Swarm Directly

Basic invoke:

```powershell
.\.venv\Scripts\python.exe -c "from langchain_core.messages import HumanMessage; from src.agent.graph import app; state={'messages':[HumanMessage(content='What is Chinchilla?')]}; print(app.invoke(state, {'configurable': {'thread_id': 'manual'}})['messages'][-1].content)"
```

The orchestrator routes to:

- `literature_reviewer`: local FAISS corpus, metadata, and DuckDuckGo
- `mathematician`: secure Python REPL with `sympy` and `numpy`
- `experimentalist`: raw PyTorch ablation scripts

Example prompts:

```text
Summarize the Chinchilla paper from the uploaded corpus.
Derive and check the gradient of x**3 + 2*x with sympy.
Write a PyTorch ablation script for comparing two transformer width settings.
```

## Agent Tools

Defined in `src/agent/tools.py`:

- `search_documentation`: searches the single local `data_lake` research corpus
- `query_research_corpus`: alias for local FAISS research search
- `query_research_metadata`: reads local corpus/index metadata
- `query_postgres_metadata`: optional read-only PostgreSQL metadata queries
- `search_web`: DuckDuckGo search for missing/current literature
- `execute_python_code`: secure math REPL with `sympy` and `numpy`
- `generate_pytorch_ablation_script`: produces raw PyTorch ablation scripts

The Python tool blocks filesystem, network, subprocess, dynamic imports, and unsafe calls such as `eval`, `exec`, `open`, and `__import__`.

## Run Tests

Focused RAG and agent tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ingestion_pipeline.py tests\test_retrieval.py tests\test_agent_tools.py tests\test_api_server.py tests\test_run_evals.py
```

Full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Evaluation

Run a real-agent evaluation sample:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.run_evals
```

Run evaluation from JSON fixtures:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.cli --dataset tests\fixtures\evaluation_dataset.json --records tests\fixtures\evaluation_records.json --output evaluation_report.json
```

## Notes

- `data_lake/source_name/...` is obsolete. Upload directly into `data_lake/`.
- `corpus/`, `rag_metadata.db`, `chroma_db/`, and `semantic_cache_db/` are generated runtime artifacts.
- PostgreSQL is optional at runtime unless you want the explicit `query_postgres_metadata` tool. Local corpus metadata works through `rag_metadata.db`.
- The old provider router demo still exists in `main.py`, but the current research-scientist path is the FastAPI/LangGraph/RAG flow described above.
