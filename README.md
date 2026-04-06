# ⚡ OmniRouter: Enterprise Agent & Async LLM Routing Engine

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Pydantic Strict](https://img.shields.io/badge/pydantic-strict-green.svg)
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


## **Second phase**
An enterprise-grade AI architecture combining high-concurrency LLM routing, local Vector Database retrieval (RAG), and LangGraph-powered state machine agents.

Originally built to solve the synchronous bottleneck of multi-provider LLM evaluations, **OmniRouter** has evolved into a complete toolkit for building autonomous, fault-tolerant research assistants capable of reading 1,000+ page technical manuals.

## ✨ Core Capabilities

* **Asynchronous Engine:** Built with `asyncio` to achieve maximum I/O throughput across OpenAI and Anthropic APIs.
* **Intelligent Failover:** Automatic exponential backoff and seamless cross-provider failover (e.g., if OpenAI rate-limits, it instantly routes to Claude 3.5).
* **Enterprise RAG:** Implements semantic chunking and local Vector Storage (`ChromaDB`) to search massive documents without blowing up context windows or API budgets.
* **Agentic Reasoning (LangGraph):** Moves beyond linear chains. Uses Directed Cyclic Graphs (DCG) and strict LLM Tool Calling to allow the agent to autonomously decide when to search databases or answer directly.
* **Strict Typing:** Powered by `Pydantic` to guardrail all API inputs and outputs against LLM hallucinations.

## 🏗 System Architecture Blueprint

```text
                      [USER PROMPT]
                            |
                            v
+=======================================================+
|                 LANGGRAPH AGENT (The Brain)           |
|                                                       |
|  [State Machine] <---> [Intent Router Node]           |
|                             |                         |
|                    (LLM Tool Calling)                 |
|                             |                         |
|  [Direct Answer] <-----> [Vector DB Search Tool]      |
+=======================================================+
                              |
                              v
+=======================================================+
|                RAG PIPELINE (The Memory)              |
|                                                       |
|  [Recursive Chunking] -> [Embeddings] -> [ChromaDB]   |
+=======================================================+
                              |
                              v
+=======================================================+
|               OMNIROUTER (The Engine)                 |
|                                                       |
|  [Pydantic Validation] -> [Async Provider Routing]    |
|  [Cost Estimation]     -> [Failover & Backoff Logic]  |
+=======================================================+


## Quick Start 2-minutes
```bash
git clone [https://github.com/YOUR_USERNAME/omnirouter.git](https://github.com/YOUR_USERNAME/omnirouter.git)
cd omnirouter
python -m venv .venv
# Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

```

## 🤝 Contributing
Please see CONTRIBUTING.md for details on adding new providers or improving the async routing logic.