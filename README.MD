<div align="center">
  <h1>🔏 Pactum</h1>
  <p><strong>AI Contract Intelligence Platform</strong></p>
  <p>A production-grade RAG system for multi-document contract analysis,<br/>built to demonstrate real AI engineering depth.</p>

  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1.3-1C3C3C?style=flat-square" />
  <img src="https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5-9B59B6?style=flat-square" />
  <img src="https://img.shields.io/badge/Phase-1%20%E2%80%94%20RAG%20MVP-27AE60?style=flat-square" />
</div>

---

<div align="center">

  <!-- Replace with your demo GIF — recommended: 800px wide, recorded at 1.5x speed -->
  <img src="assets/demo.gif" alt="Pactum Demo" width="800" />
  <br/>
  <em>Multi-document contract Q&A with cited answers and source chunk viewer</em>

</div>

---

## What is Pactum

Pactum is not a "chat with PDF" demo. It is a contract intelligence platform being built across 5 engineering phases — from a production-grade RAG MVP to hybrid retrieval, legal clause analysis, multi-agent orchestration, and graph-based reasoning.

Phase 1 establishes the core foundation: a secure, cited, multi-document RAG system with workspace isolation and a clean migration path to every subsequent phase.

---

## Phase 1 — Features

- **Multi-document workspace** — each session is isolated by UUID, persisted via URL query params, recoverable across devices
- **3-layer security** — file validation (magic bytes + extension + size), prompt injection detection, query sanitisation — built before pipeline code
- **Clause-aware chunking** — `RecursiveCharacterTextSplitter` with windowed overlap and contract-tuned separators
- **Cited answers** — every response traces to source chunks with filename, page number, and chunk index — never hallucinated
- **Source chunk viewer** — expandable inline viewer showing exact retrieved text per answer
- **Multi-turn conversation** — in-session history with character-approximate token trimming (2000 token budget)
- **Document management** — list and delete individual documents per workspace; source of truth in ChromaDB

---

## Architecture

```
User Query
    │
    ▼
Security Layer ── validate_query (injection patterns, length, type)
    │
    ▼
Retrieval Pipeline
  ├── BAAI/bge-base-en-v1.5 embeddings
  ├── ChromaDB similarity search (workspace-filtered by user_id)
  └── Pydantic Chunk → { content, file_name, page_number, chunk_index, doc_id }
    │
    ▼
Conversation Pipeline
  ├── Trim history (character approximation, 2000 token budget)
  ├── Prompt — system rules + injected context + history + query
  ├── GPT-4o-mini / Groq llama-3.3-70b-versatile (streaming)
  └── StrOutputParser
    │
    ▼
Streamlit — streamed answer + cited sources expander
```

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | GPT-4o-mini · Groq llama-3.3-70b-versatile |
| Embeddings | BAAI/bge-base-en-v1.5 · text-embedding-3-small |
| Vector DB | ChromaDB (PersistentClient) |
| RAG Framework | LangChain |
| PDF Parsing | PyPDFLoader (pypdf) |
| Validation | Pydantic |
| Security | Custom validators — magic bytes, injection patterns, query guards |

---

## Quick Start

**1. Clone and install**

```bash
git clone https://github.com/yourusername/pactum.git
cd pactum
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp .env.example .env
```

```env
ACTIVE_LLM_PROVIDER=OPENAI            # OPENAI or GROQ
ACTIVE_EMBEDDING_PROVIDER=HUGGINGFACE # HUGGINGFACE or OPENAI

OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...                  # optional — free tier via Groq
```

**3. Run**

```bash
streamlit run app.py
```

Your workspace UUID is generated on first load and embedded in the URL. Bookmark it to return to your documents.

---

## Project Structure

```
pactum/
├── app.py                    # Streamlit UI — workspace, upload, chat, citations
├── config.py                 # Provider switching, constants, paths
├── requirements.txt
├── .env.example
│
├── core/
│   ├── ingestion.py          # PDF → chunks → embeddings → ChromaDB
│   ├── document_manager.py   # list_documents, delete_document
│   ├── retrieval.py          # similarity search → Pydantic Chunk objects
│   ├── conversation.py       # history trimming, prompt, LLM call, streaming
│   └── pipeline.py           # RAG orchestrator — retrieval → conversation
│
└── security/
    └── validators.py         # validate_query, validate_file, custom exceptions
```

---

## Roadmap

| Phase | Status | Focus | Key Engineering |
|---|---|---|---|
| **1 — RAG MVP** | ✅ Shipped | Multi-document RAG | Chunking, embeddings, workspace isolation, citations, security layer |
| **2 — Hybrid RAG** | 🔨 Next | Retrieval quality | ChromaDB → Qdrant, BM25 + dense hybrid, BGE reranker, RAGAS evaluation dashboard |
| **3 — Legal Intelligence** | 📋 Planned | Domain layer | Clause classification, risk scoring, obligation extraction, plain-English summaries |
| **4 — Agentic AI** | 📋 Planned | Orchestration | LangGraph multi-agent: Retrieval, Risk, Clause, Compliance, Summary agents |
| **5 — Advanced Engineering** | 📋 Planned | Graph + memory | Neo4j graph RAG, contract diffing, temporal reasoning, long-term memory |

---

## What Makes This a Serious Engineering Project

The interesting part is not the chat interface. It is:

- **Retrieval engineering** — chunking strategy, embedding model selection, metadata schema designed from day one for the Phase 2 Qdrant migration
- **Grounded citations** — every answer traced to source chunks; the LLM never generates source metadata, the retrieval layer owns it
- **Evaluation pipeline** *(Phase 2)* — RAGAS measuring faithfulness, answer relevance, and context recall — not just qualitative vibing
- **Clean architecture** — security-first, LangChain abstractions keeping the vector store swappable, clear module boundaries with no cross-layer coupling
- **Orchestration** *(Phase 4)* — LangGraph agents with multi-step reasoning decomposed across specialised contract analysis roles

---

<div align="center">
  <sub>Built as a serious portfolio project targeting AI/ML Engineer roles — engineering depth over demo polish.</sub>
</div>