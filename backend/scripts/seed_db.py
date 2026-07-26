"""
Seed script — downloads 5 landmark GenAI papers from arXiv and ingests them.

Papers seeded:
1. "Attention Is All You Need"               (1706.03762)
2. "BERT"                                     (1810.04805)
3. "Retrieval-Augmented Generation for NLP"  (2005.11401)
4. "LoRA: Low-Rank Adaptation"               (2106.09685)
5. "Chain-of-Thought Prompting"              (2201.11903)

Usage:
    cd backend
    python -m scripts.seed_db              # normal run (skips existing papers)
    python -m scripts.seed_db --force      # wipe all papers and re-seed

Environment variables (from .env):
    DATABASE_URL, NVIDIA_API_KEY, NVIDIA_BASE_URL, EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure backend/ is on the path when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from sqlalchemy import delete, select, text

from app.core.config import get_settings
from app.core.db import AsyncSessionLocal
from app.ingestion.chunking import chunk_pages
from app.ingestion.embedder import get_embedder
from app.ingestion.loaders import load_pdf
from app.models.chunk import Chunk
from app.models.conversation import Conversation, Message
from app.models.paper import Paper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("seed_db")

# ── Paper manifest ─────────────────────────────────────────────────────────────
PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "arxiv_id": "1706.03762",
        "filename": "attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": "Devlin et al.",
        "arxiv_id": "1810.04805",
        "filename": "bert.pdf",
        "url": "https://arxiv.org/pdf/1810.04805",
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis et al.",
        "arxiv_id": "2005.11401",
        "filename": "rag_paper.pdf",
        "url": "https://arxiv.org/pdf/2005.11401",
    },
    {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al.",
        "arxiv_id": "2106.09685",
        "filename": "lora.pdf",
        "url": "https://arxiv.org/pdf/2106.09685",
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": "Wei et al.",
        "arxiv_id": "2201.11903",
        "filename": "chain_of_thought.pdf",
        "url": "https://arxiv.org/pdf/2201.11903",
    },
]

DATA_DIR = Path(__file__).parent.parent / "data" / "sample_papers"


def _download_paper(url: str, dest: Path, max_retries: int = 3) -> None:
    """Download a PDF from arXiv with retry logic."""
    if dest.exists() and dest.stat().st_size > 10_000:
        logger.info("  Already downloaded: %s", dest.name)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  Downloading %s → %s", url, dest.name)

    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(follow_redirects=True, timeout=60) as client:
                response = client.get(url)
                response.raise_for_status()
                dest.write_bytes(response.content)
            logger.info("  Downloaded %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)
            return
        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("  Download attempt %d failed: %s. Retrying in %ds...", attempt, exc, wait)
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {exc}")


async def _ingest_paper(
    db: object,
    paper_meta: dict,
    settings: object,
    chunking_strategy: str = "fixed_512",
) -> int:
    """
    Ingest a single paper: load → chunk → embed with both models → save.

    Returns the number of chunks created.
    """
    pdf_path = DATA_DIR / paper_meta["filename"]

    # Load pages
    logger.info("  Loading PDF: %s", pdf_path.name)
    pages = load_pdf(str(pdf_path))

    # Create paper row
    paper = Paper(
        title=paper_meta["title"],
        authors=paper_meta.get("authors"),
        source_filename=paper_meta["filename"],
        arxiv_id=paper_meta.get("arxiv_id"),
        total_pages=len(pages),
    )
    db.add(paper)
    await db.flush()  # get paper.id without committing the transaction
    paper_id = paper.id
    logger.info("  Paper row created (id=%d, pages=%d)", paper_id, len(pages))

    # Chunk
    chunks = chunk_pages(pages, paper_id, strategy=chunking_strategy)
    logger.info("  Chunked into %d chunks (strategy=%s)", len(chunks), chunking_strategy)

    # Embed with both models
    models_to_embed = [
        settings.embedding_model,       # nvidia/nv-embedqa-e5-v5
        settings.local_embedding_model, # BAAI/bge-m3
    ]

    total_chunks = 0
    for model_name in models_to_embed:
        logger.info("  Embedding %d chunks with %s ...", len(chunks), model_name)
        try:
            embedder = get_embedder(model_name, settings)
            texts = [c.text for c in chunks]
            vectors = embedder.embed(texts)

            chunk_rows = [
                Chunk(
                    paper_id=paper_id,
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                    chunk_text=c.text,
                    chunking_strategy=c.chunking_strategy,
                    embedding_model=model_name,
                    embedding=vec,
                    metadata_={
                        "paper_id": paper_id,
                        "paper_title": paper_meta["title"],
                        "page_number": c.page_number,
                    },
                )
                for c, vec in zip(chunks, vectors)
            ]
            db.add_all(chunk_rows)
            total_chunks += len(chunk_rows)
            logger.info("  ✓ %d chunks embedded with %s", len(chunk_rows), model_name)

        except Exception as exc:
            logger.warning("  ✗ Failed to embed with %s: %s (skipping model)", model_name, exc)

    await db.commit()
    return total_chunks


async def _seed_sample_conversations(db: object, settings: object) -> None:
    """Seed 2 sample conversations with realistic messages."""
    logger.info("Seeding sample conversations...")

    # Get some real chunk IDs for realistic sources
    from sqlalchemy import text as sql_text  # noqa: PLC0415

    result = await db.execute(
        sql_text(
            "SELECT c.id, c.page_number, p.title FROM chunks c "
            "JOIN papers p ON p.id = c.paper_id "
            "WHERE c.embedding_model = :model LIMIT 10"
        ),
        {"model": settings.embedding_model},
    )
    sample_chunks = result.mappings().all()

    if not sample_chunks:
        logger.warning("No chunks found — skipping sample conversation seed.")
        return

    def _make_sources(chunks: list) -> list[dict]:
        return [
            {
                "chunk_id": c["id"],
                "paper_title": c["title"],
                "page_number": c["page_number"],
                "score": 0.85,
                "chunk_text": "Sample chunk preview...",
            }
            for c in chunks[:3]
        ]

    conversations_data = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the attention mechanism in transformer models?",
                    "sources": None,
                },
                {
                    "role": "assistant",
                    "content": (
                        "The attention mechanism, as introduced in the Transformer architecture, "
                        "allows the model to weigh the importance of different positions in the "
                        "input sequence when producing an output. "
                        "(Source: Attention Is All You Need, p.3)"
                    ),
                    "sources": _make_sources(sample_chunks),
                    "retrieval_strategy": "hybrid",
                    "embedding_model": settings.embedding_model,
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "How does LoRA reduce the number of trainable parameters?",
                    "sources": None,
                },
                {
                    "role": "assistant",
                    "content": (
                        "LoRA freezes the pre-trained model weights and injects trainable "
                        "rank-decomposition matrices into each Transformer layer, reducing "
                        "the number of trainable parameters by orders of magnitude. "
                        "(Source: LoRA: Low-Rank Adaptation of Large Language Models, p.2)"
                    ),
                    "sources": _make_sources(sample_chunks),
                    "retrieval_strategy": "hybrid_rerank",
                    "embedding_model": settings.embedding_model,
                },
            ]
        },
    ]

    for conv_data in conversations_data:
        conv = Conversation()
        db.add(conv)
        await db.flush()

        for msg_data in conv_data["messages"]:
            msg = Message(
                conversation_id=conv.id,
                role=msg_data["role"],
                content=msg_data["content"],
                sources=msg_data.get("sources"),
                retrieval_strategy=msg_data.get("retrieval_strategy"),
                embedding_model=msg_data.get("embedding_model"),
            )
            db.add(msg)

    await db.commit()
    logger.info("✓ Seeded %d sample conversations", len(conversations_data))


async def main(force: bool = False) -> None:
    settings = get_settings()

    # Download all PDFs first
    logger.info("=== Downloading papers ===")
    for paper in PAPERS:
        _download_paper(paper["url"], DATA_DIR / paper["filename"])

    logger.info("\n=== Ingesting papers ===")
    async with AsyncSessionLocal() as db:
        if force:
            logger.warning("--force: deleting all existing papers and chunks...")
            await db.execute(delete(Paper))
            await db.commit()

        for paper_meta in PAPERS:
            logger.info("\nPaper: %s", paper_meta["title"])

            # Check if already ingested
            result = await db.execute(
                select(Paper).where(Paper.source_filename == paper_meta["filename"])
            )
            existing = result.scalar_one_or_none()
            if existing and not force:
                logger.info("  Already ingested (paper_id=%d) — skipping.", existing.id)
                continue

            try:
                n_chunks = await _ingest_paper(db, paper_meta, settings)
                logger.info("  ✓ Ingested: %d total chunk rows", n_chunks)
            except Exception as exc:
                logger.error("  ✗ Failed to ingest %s: %s", paper_meta["title"], exc)
                await db.rollback()

        # Seed sample conversations
        await _seed_sample_conversations(db, settings)

    logger.info("\n=== Seed complete ===")
    logger.info("Run 'python -m scripts.verify_seed' to verify.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the Research Paper Answer Bot database.")
    parser.add_argument("--force", action="store_true", help="Wipe and re-seed all papers.")
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
