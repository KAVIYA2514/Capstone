"""Pydantic request/response schemas for all API endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Papers ────────────────────────────────────────────────────────────────────

class PaperOut(BaseModel):
    id: int
    title: str
    authors: str | None
    source_filename: str
    arxiv_id: str | None
    total_pages: int | None
    uploaded_at: datetime
    chunk_count: int = 0

    model_config = {"from_attributes": True}


class PaperUploadResponse(BaseModel):
    paper_id: int
    title: str
    message: str
    chunks_created: int


# ── Chat ──────────────────────────────────────────────────────────────────────

class SourceOut(BaseModel):
    chunk_id: int | None
    paper_title: str
    page_number: int | None
    score: float
    chunk_text: str  # 300-char preview


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(
        None,
        description="UUID of an existing conversation. If omitted, a new one is created.",
    )
    retrieval_strategy: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid"
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What is the attention mechanism in transformer models?",
                "retrieval_strategy": "hybrid",
                "embedding_model": "nvidia/nv-embedqa-e5-v5",
            }
        }
    }


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceOut]
    retrieval_strategy: str
    embedding_model: str


# ── Conversations ─────────────────────────────────────────────────────────────

class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[dict[str, Any]] | None
    retrieval_strategy: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: uuid.UUID
    started_at: datetime
    messages: list[MessageOut] = []

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db: str
    version: str = "0.1.0"
