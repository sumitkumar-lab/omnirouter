# ⚡ OmniRouter: Enterprise RAG Agent & Async LLM Routing Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Pydantic Strict](https://img.shields.io/badge/pydantic-strict-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Streaming-009688.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-orange.svg)
![Status](https://img.shields.io/badge/status-production_ready-success.svg)

## **First phase**
A lightning-fast, highly concurrent LLM routing engine designed for production AI systems. 

When building rigorous model evaluation dashboards or running large-scale ablation studies, you cannot rely on synchronous, single-provider API calls. **OmniRouter** standardizes inputs and outputs across OpenAI, Anthropic, and local quantized models using an unbreakable Async/Abstract Base Class architecture.

## ✨ Why This Exists (The Problem)
[cite_start]Developers waste countless hours rewriting parsing logic for different LLM providers[cite: 124]. Furthermore, synchronous API calls bottleneck performance when running multi-agent workflows or bulk evaluations.

**The Solution:**
* [cite_start]**100% Asynchronous:** Built from the ground up with `asyncio` for maximum I/O throughput[cite: 72].
* **Strict Typing:** Powered by `Pydantic`. [cite_start]If a model hallucinates a bad schema, the engine catches it before it pollutes your database[cite: 73].
* [cite_start]**Provider Agnostic:** Swap between GPT-4, Claude 3, or a local Llama 3 instance with zero changes to your core application logic[cite: 102, 107, 113].

## 🏗 Architecture Blueprint

```text
[User Prompt] + [Pydantic Schema]
        |
        v
+-------------------+      +---> [OpenAI Client]    --> Returns Validated Object
|  Provider Router  | ---- |
|  (Async / Await)  | ---- +---> [Anthropic Client] --> Returns Validated Object
+-------------------+      |
                           +---> [Local VRAM Model] --> Returns Validated Object
```

## **Second phase**
An enterprise-grade AI architecture combining high-concurrency LLM routing, local Vector Database retrieval (RAG), and LangGraph-powered state machine agents.

Originally built to solve the synchronous bottleneck of multi-provider LLM evaluations, **OmniRouter** has evolved into a complete toolkit for building autonomous, fault-tolerant research assistants capable of reading 1,000+ page technical manuals.

## ✨ Core Capabilities

* **Asynchronous Engine:** Built with `asyncio` to achieve maximum I/O throughput across OpenAI and Anthropic APIs.
* **Strict Typing:** Powered by `Pydantic` to guardrail all API inputs and outputs against LLM hallucinations.
* **Intelligent Failover:** Automatic exponential backoff and seamless cross-provider failover (e.g., if OpenAI rate-limits, it instantly routes to Claude 3.5).
* **Enterprise RAG:** Implements semantic chunking and local Vector Storage (`ChromaDB`) to search massive documents without blowing up context windows or API budgets.
* **FastAPI Streaming Gateway:** Utilises Server-Sent Events (SSE) to stream tokens directly to the client, optimising Time-to-First-Token (TTFT) and eliminating perceived latency.
* **Semantic Caching Layer:** Intercepts redundant queries using HuggingFace embeddings and a secondary ChromaDB cache, returning answers in <50ms without waking up the LLM.
* **Agentic Reasoning (LangGraph):** Uses Directed Acyclic Graphs (DAG) and strict LLM Tool Calling, allowing the agent to autonomously decide when to search databases or fallback.
* **Human-in-the-Loop (HITL):** Built-in state checkpointers pause execution before dangerous tool calls, awaiting explicit asynchronous approval via the API.
* **Automated Evaluation (LLM-as-a-Judge):** Includes an automated testing pipeline that uses a secondary LLM with strict schemas to grade the primary agent on context relevance and hallucinations.


## 🏗 System Architecture Blueprint

```text
                      [CLIENT HTTP POST]
                              |
                              v
+=======================================================+
|                 FASTAPI STREAMING LAYER               |
|                                                       |
|  [Pydantic Validation] -> [Semantic Cache Check]      |
|                               /             \         |
|                     (Cache Hit)          (Cache Miss) |
|                         /                     \       |
|            [Stream Cached Answer]     [Start LangGraph]|
+=======================================================+
                                                |
                                                v
+=======================================================+
|             LANGGRAPH AGENT (The Brain)               |
|                                                       |
|  [State Machine] <---> [Intent Router / LLM]          |
|        |                        |                     |
|  (HITL Pause)          (LLM Tool Calling)             |
|        |                        |                     |
|  [Resume API] <------> [Vector DB Search Tool]        |
+=======================================================+
                                                |
                                                v
+=======================================================+
|                 RAG MEMORY PIPELINE                   |
|                                                       |
|  [HuggingFace Embeddings] -> [Local ChromaDB]         |
+=======================================================+
```
## Repository Structure...
```text
omnirouter/
├── src/
│   ├── api/                 # FastAPI server, SSE streaming, Semantic Cache
│   ├── agent/               # LangGraph state machine, nodes, and LLM tools
│   ├── evaluation/          # LLM-as-a-Judge testing scripts (run_evals.py)
│   └── rag/                 # Vector store and semantic ingestion logic
├── chroma_db/               # Primary knowledge base (git-ignored)
├── semantic_cache_db/       # Memory of past interactions (git-ignored)
├── .env                     # Secure key vault (git-ignored)
└── main.py                  # Entry point for the FastAPI server
```

## Quick Start 2-minutes

```bash
git clone [https://github.com/YOUR_USERNAME/omnirouter.git](https://github.com/YOUR_USERNAME/omnirouter.git)
cd omnirouter
python -m venv .venv
# Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

```
### Create a .env file in the root directory and add you API keys:

GROQ_API_KEY="gsk_..."

### Launch the API gateway
```bash

uvicorn src.api.server:app --reload
```
Navigate to http://127.0.0.1:8000/docs to interact with the Swagger UI, or use the included client.py to test terminal streaming.

## 🤝 Contributing
Please see CONTRIBUTING.md for details on adding new providers or improving the async routing logic.
