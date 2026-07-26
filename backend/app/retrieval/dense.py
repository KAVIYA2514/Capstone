"""
Dense cosine similarity retrieval using pgvector.

WHY cosine similarity?
For embedding models trained with cosine similarity objectives (E5, BGE),
cosine distance is the natural metric. pgvector's <=> operator computes
cosine distance (1 - cosine_similarity), so lower scores = more similar.

The query MUST filter on embedding_model to avoid mixing vectors from
different model spaces — vectors from nvidia/nv-embedqa-e5-v5 and
BAAI/bge-m3 are NOT comparable even though both are 1024-dimensional.

HNSW vs IVFFlat:
We use HNSW (Hierarchical Navigable Small World) because:
- No separate training step required (IVFFlat needs k-means clustering)
- Better recall at the same ef_search setting
- Works well for datasets up to millions of vectors
- pgvector's HNSW scales logarithmically with dataset size
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def dense_search(
    db: AsyncSession,
    query_embedding: list[float],
    embedding_model: str,
    top_k: int = 10,
    chunking_strategy: str | None = None,
    paper_ids: list[int] | None = None,
) -> list[dict]:
    """
    Retrieve the top-k chunks by cosine similarity to the query embedding.

    Args:
        db: async database session
        query_embedding: the embedded query vector (must match the column dim)
        embedding_model: which model's embeddings to search
                         ('nvidia/nv-embedqa-e5-v5' or 'BAAI/bge-m3')
        top_k: number of results to return
        chunking_strategy: optional filter (e.g. 'fixed_512', 'recursive')
        paper_ids: optional list of paper IDs to restrict search scope

    Returns:
        List of dicts with keys: chunk_id, paper_id, paper_title, page_number,
        chunk_text, chunking_strategy, embedding_model, score (cosine similarity)
    """
    # Build the embedding literal for the SQL query
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    # Cosine distance operator: 1 - (embedding <=> query) = cosine_similarity
    filters = ["c.embedding_model = :embedding_model", "c.embedding IS NOT NULL"]
    params: dict = {
        "embedding_model": embedding_model,
        "embedding": embedding_str,
        "top_k": top_k,
    }

    if chunking_strategy:
        filters.append("c.chunking_strategy = :chunking_strategy")
        params["chunking_strategy"] = chunking_strategy

    if paper_ids:
        filters.append("c.paper_id = ANY(:paper_ids)")
        params["paper_ids"] = paper_ids

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
            1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE {where_clause}
        ORDER BY c.embedding <=> CAST(:embedding AS vector) ASC
        LIMIT :top_k
    """)

    result = await db.execute(sql, params)
    rows = result.mappings().all()

    return [dict(row) for row in rows]
