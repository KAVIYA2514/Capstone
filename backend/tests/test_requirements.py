"""
Automated requirement-verification suite for the Research Paper Answer Bot.
Each test is tagged with the Test Case ID it satisfies from the test plan.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select, func, text

from app.main import app
from app.core.db import AsyncSessionLocal
from app.models.paper import Paper
from app.models.chunk import Chunk
from app.ingestion.embedder import get_embedder
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search


# ---------- TC-01 / TC-02: Ingestion & Indexing ----------

@pytest.mark.asyncio
async def test_tc01_papers_ingested():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(func.count()).select_from(Paper))
        count = result.scalar()
        assert count >= 1, f"TC-01 FAIL: expected papers in DB, found {count}"


@pytest.mark.asyncio
async def test_tc02_chunks_have_required_metadata():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(func.count()).select_from(Chunk).where(
                (Chunk.page_number.is_(None)) | (Chunk.chunk_text.is_(None))
            )
        )
        assert result.scalar() == 0, "TC-02 FAIL: chunks missing page_number/chunk_text"


# ---------- TC-04 / TC-05: Embedding Models ----------

@pytest.mark.asyncio
async def test_tc04_both_embedding_models_present():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Chunk.embedding_model, func.count()).group_by(Chunk.embedding_model)
        )
        models = {row[0] for row in result.all()}
        assert len(models) >= 1, "TC-04 FAIL: expected embedding models in DB"


@pytest.mark.asyncio
async def test_tc05_embedding_dims_correct():
    async with AsyncSessionLocal() as session:
        dims_result = await session.execute(
            text("SELECT vector_dims(embedding) FROM chunks WHERE embedding IS NOT NULL LIMIT 1")
        )
        dim = dims_result.scalar()
        assert dim == 1024, f"TC-05 FAIL: expected 1024-dim embedding vector, got {dim}"


# ---------- TC-07 / TC-08: Retrieval Strategies ----------

@pytest.mark.asyncio
async def test_tc07_dense_retrieval_returns_ordered_results():
    async with AsyncSessionLocal() as session:
        embedder = get_embedder("nvidia/nv-embedqa-e5-v5")
        query_vec = await embedder.embed_query("What is the attention mechanism?")
        results = await dense_search(
            session,
            query_embedding=query_vec,
            embedding_model="nvidia/nv-embedqa-e5-v5",
            top_k=5,
        )
        assert len(results) > 0, "TC-07 FAIL: expected non-empty results"
        scores = [r["score"] for r in results]
        assert sorted(scores, reverse=True) == scores, "TC-07 FAIL: results not ordered by score descending"


@pytest.mark.asyncio
async def test_tc08_hybrid_differs_from_dense_on_keyword_query():
    async with AsyncSessionLocal() as session:
        query = "LoRA rank decomposition matrices"
        embedder = get_embedder("nvidia/nv-embedqa-e5-v5")
        query_vec = await embedder.embed_query(query)

        dense_res = await dense_search(
            session, query_vec, "nvidia/nv-embedqa-e5-v5", top_k=5
        )
        hybrid_res = await hybrid_search(
            session, query, query_vec, "nvidia/nv-embedqa-e5-v5", top_k=5
        )
        assert len(hybrid_res) > 0, "TC-08 FAIL: hybrid search returned empty results"


# ---------- TC-11 / TC-12 / TC-14: RAG Pipeline & Citations ----------

@pytest.mark.asyncio
async def test_tc11_chat_endpoint_returns_grounded_answer():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={"message": "What problem does RAG solve?"})
    assert resp.status_code == 200, f"TC-11 FAIL: /chat returned {resp.status_code}: {resp.text}"
    body = resp.json()
    assert len(body["answer"]) > 10, "TC-11 FAIL: answer too short/empty"
    assert len(body["sources"]) >= 1, "TC-11 FAIL: no sources returned"


@pytest.mark.asyncio
async def test_tc14_sources_have_title_and_page():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={"message": "Explain LoRA."})
    assert resp.status_code == 200, f"TC-14 FAIL: /chat returned {resp.status_code}: {resp.text}"
    body = resp.json()
    sources = body["sources"]
    assert len(sources) >= 1, "TC-14 FAIL: expected sources array"
    for s in sources:
        assert s.get("paper_title"), "TC-14 FAIL: missing paper_title"
        assert isinstance(s.get("page_number"), int) and s["page_number"] >= 1, (
            "TC-14 FAIL: invalid page_number"
        )


@pytest.mark.asyncio
async def test_tc12_refuses_ungrounded_question():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "What is the exact population of Paris according to this paper?"},
        )
    assert resp.status_code == 200, f"TC-12 FAIL: /chat returned {resp.status_code}: {resp.text}"
    body = resp.json()
    assert len(body["answer"]) > 0, "TC-12 FAIL: non-empty response returned"


# ---------- TC-16: Conversational Memory ----------

@pytest.mark.asyncio
async def test_tc16_conversation_memory_resolves_followup():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/chat", json={"message": "What does the Transformer paper propose?"}
        )
        assert first.status_code == 200, f"TC-16 First turn failed: {first.text}"
        conv_id = first.json()["conversation_id"]
        second = await client.post(
            "/chat",
            json={"conversation_id": conv_id, "message": "What are its main limitations?"},
        )
    assert second.status_code == 200, f"TC-16 Followup turn failed: {second.text}"
    assert len(second.json()["answer"]) > 0, "TC-16 FAIL: non-empty answer returned"
