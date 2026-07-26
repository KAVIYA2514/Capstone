# 🧪 Research Paper Answer Bot — Complete Test Plan & Traceability Matrix

### Traceability: Every test case maps to a specific line item in the Capstone brief / rubric

---

## 1. Traceability Matrix

| Brief Requirement | Test Case(s) | Rubric Marks | Status |
|---|---|---|---|
| Compulsory Goal 1 — dataset/PDFs collected | TC-01 | Problem Understanding (15) | ✅ PASS |
| Compulsory Goal 2 — load & index in vector DB | TC-02, TC-03 | Problem Understanding (15) | ✅ PASS |
| Compulsory Goal 3 — ≥2 embedding models compared | TC-04, TC-05, TC-06 | Vector DB & Embeddings (15) | ✅ PASS |
| Compulsory Goal 4 — multiple retrieval strategies | TC-07, TC-08, TC-09, TC-10 | Retrieval Strategy (15) | ✅ PASS |
| Compulsory Goal 5 — RAG pipeline built | TC-11, TC-12 | RAG Pipeline (15) | ✅ PASS |
| Compulsory Goal 6 — tested on sample queries | TC-13 | RAG Pipeline (15) | ✅ PASS |
| Compulsory Goal 7 — top-3 sources w/ title+page | TC-14, TC-15 | RAG Pipeline (15) | ✅ PASS |
| Stretch Goal — at least 1 implemented | TC-16 (memory), TC-17 (UI) | Bonus | ✅ PASS |
| Deliverable — code runs end-to-end | TC-18 | Python (10) | ✅ PASS |
| Deliverable — experiments documented inline | TC-19 | Technical Understanding | ✅ PASS |
| Deliverable — presentation complete | TC-20 | N/A (manual) | ✅ READY |
| Deliverable — live demo, 5 queries | TC-21 | N/A (manual) | ✅ READY |
| Prompt engineering — grounding, refusal | TC-22, TC-23 | Prompt Engineering (10) | ✅ PASS |
| RAG concept — precision/recall trade-off documented | TC-24 | RAG (10) | ✅ PASS |
| LangChain component usage | TC-25 | LangChain (10) | ✅ PASS |

---

## 2. Test Case Execution Details

### 📥 Data Ingestion & Indexing (Compulsory Goals 1–2)
- **TC-01 — Papers collected and loadable**
  - **Pass criteria**: Seed script ingests ≥ 5 landmark papers without failure.
- **TC-02 — Documents chunked and indexed with metadata**
  - **Pass criteria**: Chunks possess non-null `paper_id`, `page_number`, and `chunk_text`.
- **TC-03 — Edge case: scanned/low-text PDF handling**
  - **Pass criteria**: OCR / page loader logs fallback warning without crashing process.

### 🧠 Embedding Models (Compulsory Goal 3)
- **TC-04 — Two embedding models produce vectors**
  - **Pass criteria**: Chunks indexed for both `nvidia/nv-embedqa-e5-v5` and `BAAI/bge-m3`.
- **TC-05 — Embedding dimensionality correct**
  - **Pass criteria**: `pgvector` column stores 1024-dimensional float vectors.
- **TC-06 — Qualitative embedding comparison documented**
  - **Pass criteria**: Comparative analysis present in notebook (`notebooks/experiments.ipynb`).

### 🔍 Retrieval Strategies (Compulsory Goal 4)
- **TC-07 — Dense (cosine) retrieval returns relevant chunks**
  - **Pass criteria**: Returns ordered candidates by Cosine similarity distance.
- **TC-08 — Hybrid retrieval combines dense + keyword**
  - **Pass criteria**: Full-text BM25 (`tsvector`) + Dense Vector Search fused via Reciprocal Rank Fusion (RRF).
- **TC-09 — Reranker (Cross-Encoder) optimizes top results**
  - **Pass criteria**: `ms-marco-MiniLM-L-6-v2` re-ranks candidates for high precision.
- **TC-10 — Retrieval strategies compared with documented winner**
  - **Pass criteria**: Comparative experiments recorded with rationale.

### 🤖 RAG Pipeline & Generation (Compulsory Goals 5–7)
- **TC-11 — RAG chain connects retriever to LLM**
  - **Pass criteria**: `/chat` endpoint returns streaming answer + sources payload.
- **TC-12 — Grounding & refusal mechanism**
  - **Pass criteria**: Returns graceful "insufficient context" message when query is ungrounded.
- **TC-13 — Representative sample query set**
  - **Pass criteria**: Batch queries pass without unhandled HTTP exceptions.
- **TC-14 — Top-3 sources with title + page number**
  - **Pass criteria**: Returns top 1–3 citations with exact paper titles and page numbers.
- **TC-15 — Frontend renders citations correctly**
  - **Pass criteria**: UI drawer renders citations and chunk text previews dynamically.

---

## 3. Automated Test Suite Integration

The automated verification suite has been created in:
- **[test_requirements.py](file:///d:/Projects/Capstone/backend/tests/test_requirements.py)**
- **[conftest.py](file:///d:/Projects/Capstone/backend/tests/conftest.py)**

To execute the test suite:
```powershell
cd backend
.venv\Scripts\activate
pytest tests/test_requirements.py -v
```
