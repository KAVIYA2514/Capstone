"""
FastAPI application entrypoint.

Startup order:
1. Settings are validated (pydantic-settings reads .env)
2. DB engine is created (connection pool is established)
3. CORS middleware is registered
4. Routers are mounted
5. The app is ready to serve requests

Note: We do NOT auto-create tables here. Run Alembic migrations:
    cd backend && alembic upgrade head
"""

from __future__ import annotations

import asyncio
import logging
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import health, papers, chat


# Configure logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Research Paper Answer Bot",
    description=(
        "RAG chatbot over GenAI/LLM research papers. "
        "Supports dense, hybrid, and hybrid+rerank retrieval strategies. "
        "Embedding model comparison: nvidia/nv-embedqa-e5-v5 vs BAAI/bge-m3."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(papers.router)
app.include_router(chat.router)


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "message": "Research Paper Answer Bot API",
        "docs": "/docs",
        "health": "/health",
    }
