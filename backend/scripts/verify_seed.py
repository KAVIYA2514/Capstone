"""
Verify that the seed script ran correctly.

Prints a summary table:
  Paper title | Pages | Chunks (fixed_512) | Chunks (recursive) | Chunks (nvidia) | Chunks (bge-m3)

Usage:
    cd backend
    python -m scripts.verify_seed
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func, select, text

from app.core.db import AsyncSessionLocal
from app.models.chunk import Chunk
from app.models.conversation import Conversation, Message
from app.models.paper import Paper


async def main() -> None:
    async with AsyncSessionLocal() as db:
        # ── Papers summary ──────────────────────────────────────────────────
        result = await db.execute(
            select(
                Paper.id,
                Paper.title,
                Paper.total_pages,
                func.count(Chunk.id).label("total_chunks"),
            )
            .outerjoin(Chunk, Chunk.paper_id == Paper.id)
            .group_by(Paper.id)
            .order_by(Paper.id)
        )
        papers = result.all()

        if not papers:
            print("No papers found in DB. Run: python -m scripts.seed_db")
            return

        print("\n" + "=" * 100)
        print("SEED VERIFICATION REPORT")
        print("=" * 100)

        print(f"\n{'Paper':<50} {'Pages':>5} {'Total Chunks':>12}")
        print("-" * 70)
        for paper in papers:
            print(f"{paper.title[:48]:<50} {paper.total_pages or '?':>5} {paper.total_chunks:>12}")

        # ── Chunks by strategy ───────────────────────────────────────────────
        print("\n\nChunks by Strategy:")
        strat_result = await db.execute(
            select(
                Paper.title,
                Chunk.chunking_strategy,
                func.count(Chunk.id).label("n"),
            )
            .join(Chunk, Chunk.paper_id == Paper.id)
            .group_by(Paper.title, Chunk.chunking_strategy)
            .order_by(Paper.title, Chunk.chunking_strategy)
        )
        strat_rows = strat_result.all()

        print(f"\n{'Paper':<50} {'Strategy':<15} {'Count':>8}")
        print("-" * 76)
        for row in strat_rows:
            print(f"{row.title[:48]:<50} {row.chunking_strategy:<15} {row.n:>8}")

        # ── Chunks by embedding model ────────────────────────────────────────
        print("\n\nChunks by Embedding Model:")
        model_result = await db.execute(
            select(
                Paper.title,
                Chunk.embedding_model,
                func.count(Chunk.id).label("n"),
            )
            .join(Chunk, Chunk.paper_id == Paper.id)
            .group_by(Paper.title, Chunk.embedding_model)
            .order_by(Paper.title, Chunk.embedding_model)
        )
        model_rows = model_result.all()

        print(f"\n{'Paper':<50} {'Embedding Model':<35} {'Count':>8}")
        print("-" * 96)
        for row in model_rows:
            print(f"{row.title[:48]:<50} {row.embedding_model:<35} {row.n:>8}")

        # ── Conversations ────────────────────────────────────────────────────
        conv_result = await db.execute(select(func.count(Conversation.id)))
        n_convs = conv_result.scalar()
        msg_result = await db.execute(select(func.count(Message.id)))
        n_msgs = msg_result.scalar()

        print(f"\n\nConversations: {n_convs}  |  Messages: {n_msgs}")

        # ── Vector index check ───────────────────────────────────────────────
        idx_result = await db.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename = 'chunks' ORDER BY indexname"
            )
        )
        indexes = idx_result.all()
        print("\n\nActive indexes on 'chunks' table:")
        for idx in indexes:
            print(f"  {idx.indexname}")

        print("\n" + "=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
