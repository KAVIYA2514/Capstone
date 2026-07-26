"""
Chunk ORM model.

Each row represents one text chunk from a paper, embedded with one model.
To compare two embedding models (nvidia/nv-embedqa-e5-v5 vs BAAI/bge-m3),
we insert *separate rows per model* — the embedding_model column identifies
which model produced the vector.

Why one vector(1024) column instead of two separate columns?
Both nvidia/nv-embedqa-e5-v5 and BAAI/bge-m3 output 1024-dimensional vectors,
so a single typed column suffices. This is cleaner than the two-column design
that would be required if the dimensionalities differed (e.g., 768 vs 1024).
The trade-off is that retrieval queries MUST filter by embedding_model to avoid
mixing embeddings from different model spaces.

The tsv column is a Postgres GENERATED ALWAYS AS column (computed server-side)
that stores a pre-built tsvector for fast full-text search. It's populated
automatically on INSERT/UPDATE — no application code needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Which chunking strategy produced this chunk:
    # 'fixed_512' | 'recursive' | 'semantic'
    chunking_strategy: Mapped[str] = mapped_column(String(50), nullable=False)

    # Which embedding model produced the vector:
    # 'nvidia/nv-embedqa-e5-v5' | 'BAAI/bge-m3'
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)

    # Single 1024-dim embedding column — works for both models since both
    # output 1024-dimensional vectors. Queries MUST filter on embedding_model
    # to stay within a single model's vector space.
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=True)

    # Pre-built tsvector for full-text search (populated by Postgres trigger
    # or GENERATED column in migration)
    tsv: Mapped[Any] = mapped_column(TSVECTOR, nullable=True)

    # Flexible metadata sidecar: paper_title, page_number, chunk_index, etc.
    # Stored here so retrieval results are self-contained without joins.
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="'{}'::jsonb",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    paper: Mapped["Paper"] = relationship("Paper", back_populates="chunks")  # noqa: F821

    __table_args__ = (
        # HNSW index for approximate nearest-neighbor search on embedding vectors.
        # vector_cosine_ops uses cosine distance (1 - cosine_similarity),
        # which is what the pgvector <=> operator measures.
        # m=16, ef_construction=64 are sensible defaults for a research project;
        # increase ef_construction to 128+ for production precision.
        Index(
            "idx_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN index on tsvector for O(log n) full-text search.
        Index("idx_chunks_tsv_gin", "tsv", postgresql_using="gin"),
        # B-tree indexes for common filter columns
        Index("idx_chunks_paper_id", "paper_id"),
        Index("idx_chunks_embedding_model", "embedding_model"),
        Index("idx_chunks_chunking_strategy", "chunking_strategy"),
    )

    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.id} paper_id={self.paper_id} "
            f"page={self.page_number} model={self.embedding_model!r}>"
        )
