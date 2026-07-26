# Research Paper Answer Bot

A production-quality RAG chatbot over GenAI/LLM research papers.

**Stack**: FastAPI + PostgreSQL/pgvector (Neon) + React + NVIDIA NIM

---

## Quick Start (Local Dev — No Docker)

### Prerequisites
- Python 3.11+
- Node.js 18+
- A [Neon](https://neon.tech) PostgreSQL database (free tier works)
- NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com)

### 1. Clone & configure

```bash
cd d:/Projects/Capstone
cp .env.example .env
# Edit .env and fill in:
#   DATABASE_URL=postgresql+asyncpg://...@...neon.tech/...?sslmode=require
#   NVIDIA_API_KEY=nvapi-...
```

### 2. Backend setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -e ".[dev]"

# Run database migrations (creates all tables + indexes on Neon)
alembic upgrade head

# Start the API server
uvicorn app.main:app --reload --port 8000
```

API will be at **http://localhost:8000**
Swagger docs at **http://localhost:8000/docs**

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend will be at **http://localhost:5173**

### 4. Seed the database (optional but recommended)

```bash
# In the backend/ directory, with venv activated:
python -m scripts.seed_db

# Verify:
python -m scripts.verify_seed
```

This downloads 5 landmark GenAI papers from arXiv and ingests them with both embedding models (~5–10 min depending on network and GPU).

---

## Architecture

```
User ──► React (Vite, TanStack Query, Tailwind)
           │
           ▼  HTTP
FastAPI Backend
 ├── POST /papers/upload  →  PDF loaders → chunker → embedder → Neon
 ├── POST /chat           →  embed query → retriever → LLM → response
 └── GET  /papers         →  list papers + chunk counts
           │
           ├── Neon PostgreSQL + pgvector
           │    ├── papers, chunks (vector(1024))
           │    └── conversations, messages
           │
           ├── Embedding Models
           │    ├── nvidia/nv-embedqa-e5-v5  (NVIDIA API)
           │    └── BAAI/bge-m3              (local sentence-transformers)
           │
           ├── Retrieval Strategies
           │    ├── Dense: pgvector <=> cosine
           │    ├── Hybrid: Dense + ts_rank via RRF
           │    └── Hybrid+Rerank: Hybrid + CrossEncoder ms-marco-MiniLM
           │
           └── LLM
                ├── meta/llama-4-maverick-17b-128e-instruct  (NVIDIA NIM)
                └── llama3.1:8b (Ollama fallback)
```

---

## Running Experiments (Notebook)

```bash
cd backend
# Start Jupyter
jupyter notebook ../notebooks/experiments.ipynb
```

The notebook documents:
1. Chunking strategy comparison (fixed_512 / recursive / semantic)
2. Embedding model comparison (nvidia/nv-embedqa-e5-v5 vs BAAI/bge-m3)
3. Retrieval strategy comparison (dense / hybrid / hybrid+rerank)

---

## Running the Retrieval Evaluator

```bash
cd backend
python -m app.retrieval.evaluator
# Output: notebooks/retrieval_eval_results.md
```

## Running RAGAS Evaluation

```bash
cd backend
python -m app.evaluation.ragas_eval
```

---

## Project Structure

```
research-paper-answer-bot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── core/                # config, db
│   │   ├── models/              # SQLAlchemy ORM
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── ingestion/           # PDF loaders, chunkers, embedders
│   │   ├── retrieval/           # dense, hybrid, reranker, evaluator
│   │   ├── rag/                 # LCEL chain, memory
│   │   ├── api/routes/          # FastAPI routes
│   │   └── evaluation/          # RAGAS eval
│   ├── alembic/                 # DB migrations
│   ├── scripts/                 # seed_db, verify_seed
│   └── pyproject.toml
├── frontend/
│   └── src/
│       ├── api/client.ts        # typed API client
│       ├── components/          # ChatWindow, MessageBubble, SourceCitations, ...
│       ├── pages/ChatPage.tsx
│       └── hooks/useChatSession.ts
├── notebooks/experiments.ipynb
├── docs/architecture-diagram.md
└── .env.example
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | Neon asyncpg connection string |
| `NVIDIA_API_KEY` | ✅ | NVIDIA NIM API key (`nvapi-...`) |
| `NVIDIA_BASE_URL` | ✅ | `https://integrate.api.nvidia.com/v1` |
| `EMBEDDING_MODEL` | ✅ | `nvidia/nv-embedqa-e5-v5` |
| `EMBEDDING_DIM` | ✅ | `1024` |
| `CHAT_MODEL` | ✅ | `meta/llama-4-maverick-17b-128e-instruct` |
| `LLM_PROVIDER` | — | `nvidia_nim` or `ollama` (default: `nvidia_nim`) |
| `LOCAL_EMBEDDING_MODEL` | — | `BAAI/bge-m3` (comparison model) |
| `CORS_ORIGINS` | — | Frontend origin(s) |
