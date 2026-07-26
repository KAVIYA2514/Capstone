"""
Papers API routes — upload PDFs and list ingested papers.

Ingestion runs as a FastAPI BackgroundTask so the upload endpoint returns
immediately with a paper_id while the PDF is being processed asynchronously.
The caller can poll GET /papers/{paper_id} to check status (or just list
papers after a reasonable delay).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import get_db
from app.ingestion.chunking import chunk_pages
from app.ingestion.embedder import get_embedder
from app.ingestion.loaders import load_pdf
from app.models.chunk import Chunk
from app.models.paper import Paper
from app.schemas.schemas import PaperOut, PaperUploadResponse

router = APIRouter(prefix="/papers", tags=["papers"])
logger = logging.getLogger(__name__)


async def _ingest_paper(
    paper_id: int,
    file_path: str,
    chunking_strategy: str = "recursive",
) -> None:
    """
    Background ingestion task: load PDF → chunk → embed (both models) → save.

    We embed with BOTH models so retrieval comparisons work immediately
    without re-ingesting. The chunking strategy defaults to 'recursive'
    for uploads, but can be overridden.
    """
    from app.core.db import AsyncSessionLocal  # noqa: PLC0415

    settings = get_settings()

    async with AsyncSessionLocal() as db:
        try:
            # Load pages
            pages = load_pdf(file_path)

            # Chunk
            chunks = chunk_pages(pages, paper_id, strategy=chunking_strategy)
            logger.info("Ingestion: %d chunks from paper_id=%d", len(chunks), paper_id)

            # Embed with both models
            models_to_embed = [
                settings.embedding_model,         # nvidia/nv-embedqa-e5-v5
                settings.local_embedding_model,   # BAAI/bge-m3
            ]

            for model_name in models_to_embed:
                logger.info("Embedding with model: %s", model_name)
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
                            **c.metadata,
                            "paper_id": paper_id,
                            "page_number": c.page_number,
                        },
                    )
                    for c, vec in zip(chunks, vectors)
                ]
                db.add_all(chunk_rows)
                await db.commit()
                logger.info(
                    "Saved %d chunk embeddings (model=%s)", len(chunk_rows), model_name
                )

            # Update paper total_pages
            paper_result = await db.execute(select(Paper).where(Paper.id == paper_id))
            paper = paper_result.scalar_one()
            paper.total_pages = len(pages)
            await db.commit()

        except Exception as exc:
            logger.error("Ingestion failed for paper_id=%d: %s", paper_id, exc)
            await db.rollback()


@router.post("/upload", response_model=PaperUploadResponse, status_code=201)
async def upload_paper(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    chunking_strategy: str = "recursive",
    db: AsyncSession = Depends(get_db),
) -> PaperUploadResponse:
    """
    Upload a PDF and trigger asynchronous ingestion.

    The paper row is created immediately (so paper_id is returned to the
    caller). Chunking and embedding run in the background.

    Args:
        file: the PDF file (multipart/form-data)
        chunking_strategy: 'fixed_512' | 'recursive' | 'semantic'
        background_tasks: FastAPI background task runner
        db: async database session
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted.")

    # Check for duplicate
    existing = await db.execute(
        select(Paper).where(Paper.source_filename == file.filename)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail=f"Paper '{file.filename}' already ingested. "
                   "Use DELETE /papers/{id} first to re-ingest.",
        )

    # Save to temp file (we need a file path for the PDF loaders)
    content = await file.read()
    if len(content) < 100:
        raise HTTPException(status_code=422, detail="Uploaded file is too small or empty.")

    tmp_path = tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False, dir=tempfile.gettempdir()
    )
    tmp_path.write(content)
    tmp_path.close()

    # Infer title from filename (will be updated from PDF metadata if available)
    title = Path(file.filename).stem.replace("_", " ").replace("-", " ").title()

    # Create the paper row immediately
    paper = Paper(title=title, source_filename=file.filename)
    db.add(paper)
    await db.commit()
    await db.refresh(paper)

    # Schedule background ingestion
    background_tasks.add_task(
        _ingest_paper,
        paper_id=paper.id,
        file_path=tmp_path.name,
        chunking_strategy=chunking_strategy,
    )

    return PaperUploadResponse(
        paper_id=paper.id,
        title=paper.title,
        message="Ingestion started. Chunks will be available in ~30-60s.",
        chunks_created=0,  # not yet available — ingestion is async
    )


@router.get("", response_model=list[PaperOut])
async def list_papers(db: AsyncSession = Depends(get_db)) -> list[PaperOut]:
    """List all ingested papers with their chunk counts."""
    result = await db.execute(
        select(
            Paper,
            func.count(Chunk.id).label("chunk_count"),
        )
        .outerjoin(Chunk, Chunk.paper_id == Paper.id)
        .group_by(Paper.id)
        .order_by(Paper.uploaded_at.desc())
    )
    rows = result.all()

    papers = []
    for paper, chunk_count in rows:
        po = PaperOut.model_validate(paper)
        po.chunk_count = chunk_count or 0
        papers.append(po)

    return papers


@router.get("/{paper_id}", response_model=PaperOut)
async def get_paper(paper_id: int, db: AsyncSession = Depends(get_db)) -> PaperOut:
    """Get a single paper by ID."""
    result = await db.execute(
        select(Paper, func.count(Chunk.id).label("chunk_count"))
        .outerjoin(Chunk, Chunk.paper_id == Paper.id)
        .where(Paper.id == paper_id)
        .group_by(Paper.id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found.")
    paper, chunk_count = row
    po = PaperOut.model_validate(paper)
    po.chunk_count = chunk_count or 0
    return po


@router.delete("/{paper_id}", status_code=204)
async def delete_paper(paper_id: int, db: AsyncSession = Depends(get_db)) -> None:
    """Delete a paper and all its chunks (CASCADE)."""
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found.")
    await db.delete(paper)
    await db.commit()
