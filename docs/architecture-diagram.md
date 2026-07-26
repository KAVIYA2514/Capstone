# Architecture Diagram — Research Paper Answer Bot

## End-to-End Architecture

```mermaid
flowchart TD
    subgraph Frontend["Frontend (React + Vite)"]
        UI["Chat UI\nChatWindow + MessageBubble"]
        SC["SourceCitations\n(paper title + page)"]
        DP["DevPanel\n(strategy + model selector)"]
        PU["PaperUploader\n(drag-and-drop PDF)"]
        PL["PaperLibrary\n(ingested papers list)"]
    end

    subgraph Backend["Backend (FastAPI)"]
        API["REST API\n/chat · /papers · /health"]

        subgraph Ingestion["Ingestion Pipeline"]
            L["PDF Loaders\npypdf → pdfplumber → OCR"]
            C["Chunkers\nfixed_512 · recursive · semantic"]
            E["Embedders\nNVIDIA E5-v5 · BAAI/bge-m3"]
        end

        subgraph Retrieval["Retrieval Layer"]
            D["Dense\npgvector <=> cosine"]
            H["Hybrid RRF\nDense + ts_rank"]
            R["Reranker\nms-marco CrossEncoder"]
        end

        subgraph RAG["RAG Chain"]
            QE["Query Embedder"]
            CH["Chat History\n(last N turns from DB)"]
            LLM["LLM\nNVIDIA NIM / Ollama"]
        end
    end

    subgraph DB["Neon PostgreSQL + pgvector"]
        T1["papers"]
        T2["chunks\n(vector 1024-dim)\nHNSW index + GIN index"]
        T3["conversations · messages\n(JSONB sources)"]
    end

    subgraph External["External APIs"]
        NIM["NVIDIA NIM\nmeta/llama-4-maverick\nnv-embedqa-e5-v5"]
    end

    %% User flow
    UI -->|POST /chat| API
    DP -->|retrieval_strategy\nembedding_model| API
    PU -->|POST /papers/upload| API

    %% Ingestion
    API --> L --> C --> E --> T2

    %% Retrieval
    API --> QE --> D & H
    H --> R
    D & R --> LLM

    %% Memory
    T3 --> CH --> LLM

    %% LLM
    LLM -->|answer + sources| API

    %% DB connections
    D & H --> T2
    API --> T1 & T3

    %% External
    E & LLM --> NIM

    %% Response
    API -->|answer + citations| SC
    SC --> UI
```

## Component Summary

| Component | Technology | Purpose |
|---|---|---|
| **Frontend** | React 18 + Vite + Tailwind | Chat UI, paper management, dev panel |
| **API** | FastAPI + Pydantic | REST endpoints, validation, CORS |
| **Ingestion** | pypdf + pdfplumber + tiktoken | PDF → pages → chunks |
| **Embedders** | NVIDIA API + sentence-transformers | Dual-model embedding for comparison |
| **Vector Store** | Neon PostgreSQL + pgvector | Dense cosine search (HNSW) |
| **Full-text** | Postgres tsvector + GIN | BM25-equivalent keyword search |
| **Retrieval** | Custom Python | Dense / Hybrid RRF / CrossEncoder rerank |
| **LLM** | NVIDIA NIM (ChatOpenAI) | Answer generation |
| **Memory** | PostgreSQL messages table | Conversational history |

## Why PostgreSQL over Chroma/FAISS?

1. **One data store** — relational + vector + full-text in a single system. No sync between a metadata store and a separate vector index.
2. **Native hybrid search** — pgvector `<=>` + `ts_rank` combined in a single SQL query via RRF. No external BM25 server needed.
3. **Production-realistic** — Neon is a serverless Postgres; this architecture scales to production without replacing any component.
4. **Auditable** — every query, answer, retrieval strategy, and embedding model is logged in the `messages` table as JSONB for reproducibility.
