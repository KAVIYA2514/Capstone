"""
Hybrid retrieval: pgvector dense search + Postgres full-text search,
fused with Reciprocal Rank Fusion (RRF).

WHY hybrid?
Pure dense retrieval can miss exact keyword matches (acronyms, proper nouns,
model names like "GPT-4", "BERT", "LoRA"). Pure BM25/full-text search misses
semantically similar but lexically different passages. Combining both gives:
- Recall: captures both semantic and exact-match relevance
- Precision: RRF promotes results that rank well on *both* signals

WHY Reciprocal Rank Fusion?
RRF(d) = Σ 1 / (k + rank_i) where k=60 is a smoothing constant.
Compared to a simple weighted linear combination of scores, RRF is:
- Score-scale agnostic (cosine similarities vs ts_rank scores have different
  scales; RRF only uses rank positions, not raw values)
- More robust to outlier scores
- Parameter-light (just k, not a separately tuned alpha weight)

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"

Implementation note:
We run both searches in parallel (asyncio.gather), then fuse in Python.
An alternative is a single SQL query with UNION + window functions, but
the two-query approach is easier to extend (e.g., adding a third signal
like BM25 from a separate index).
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.dense import dense_search

logger = logging.getLogger(__name__)

# RRF smoothing constant — 60 is the canonical value from the original paper
_RRF_K = 60


async def fulltext_search(
    db: AsyncSession,
    query: str,
    embedding_model: str,
    top_k: int = 20,
    chunking_strategy: str | None = None,
) -> list[dict]:
    """
    Full-text search using Postgres tsvector / ts_rank.

    Uses the GIN-indexed tsv column (auto-populated by trigger on insert).
    plainto_tsquery is used instead of to_tsquery because it doesn't require
    the caller to add & / | operators — it handles natural language queries.

    The embedding_model filter is included so we don't return the same chunk
    text twice (once per model) in the fused results.
    """
    filters = [
        "c.tsv @@ plainto_tsquery('english', :query)",
        "c.embedding_model = :embedding_model",
    ]
    params: dict = {
        "query": query,
        "embedding_model": embedding_model,
        "top_k": top_k,
    }
    if chunking_strategy:
        filters.append("c.chunking_strategy = :chunking_strategy")
        params["chunking_strategy"] = chunking_strategy

    where_clause = " AND ".join(filters)

    sql = text(f"""
        SELECT
            c.id            AS chunk_id,
            c.paper_id,
            p.title         AS paper_title,
            c.page_number,
            c.chunk_text,
            c.chunking_strategy,
            c.embedding_model,
            ts_rank(c.tsv, plainto_tsquery('english', :query)) AS score
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE {where_clause}
        ORDER BY score DESC
        LIMIT :top_k
    """)

    result = await db.execute(sql, params)
    rows = result.mappings().all()
    return [dict(row) for row in rows]


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = _RRF_K,
) -> list[dict]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: list of ranked result lists (each a list of dicts with chunk_id)
        k: smoothing constant (default 60)

    Returns:
        Single merged list sorted by descending RRF score, with an added
        'rrf_score' key and a 'score' key reflecting the fused relevance.
    """
    scores: dict[int, float] = {}
    chunk_data: dict[int, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = item

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for chunk_id, rrf_score in merged:
        entry = dict(chunk_data[chunk_id])
        entry["score"] = rrf_score
        entry["rrf_score"] = rrf_score
        results.append(entry)

    return results


async def hybrid_search(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    embedding_model: str,
    top_k: int = 20,
    chunking_strategy: str | None = None,
    paper_ids: list[int] | None = None,
) -> list[dict]:
    """
    Hybrid retrieval combining dense vector search and BM25-equivalent full-text
    search via Reciprocal Rank Fusion.

    Args:
        db: async database session
        query: raw query string (for full-text search)
        query_embedding: embedded query vector (for dense search)
        embedding_model: which model's embedding column to search
        top_k: number of candidates to fetch from each sub-retriever
        chunking_strategy: optional filter
        paper_ids: optional paper scope filter

    Returns:
        Merged and RRF-reranked list of chunks (up to top_k results)
    """
    # Run both retrievers sequentially to avoid asyncpg connection state conflicts
    dense_results = await dense_search(
        db,
        query_embedding,
        embedding_model,
        top_k=top_k,
        chunking_strategy=chunking_strategy,
        paper_ids=paper_ids,
    )
    ft_results = await fulltext_search(
        db,
        query,
        embedding_model,
        top_k=top_k,
        chunking_strategy=chunking_strategy,
    )

    logger.debug(
        "hybrid_search: dense=%d ft=%d candidates",
        len(dense_results),
        len(ft_results),
    )

    fused = _reciprocal_rank_fusion([dense_results, ft_results])
    return fused[:top_k]
