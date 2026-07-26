"""
Chat API route — the main RAG endpoint.

POST /chat:
  - Creates or continues a conversation
  - Embeds the user query using the selected model
  - Retrieves relevant chunks using the selected strategy
  - Generates a grounded answer using the LLM
  - Persists both the user message and the assistant response with sources
  - Returns the answer + top-3 source citations

GET /conversations/{id}:
  - Returns the full conversation history
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import get_db
from app.rag.chain import rag_query
from app.rag.memory import (
    get_chat_history,
    get_or_create_conversation,
    save_message,
)
from app.schemas.schemas import ChatRequest, ChatResponse, ConversationOut, SourceOut

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Main RAG chat endpoint.

    Accepts a question and optional conversation context, retrieves relevant
    chunks from the vector store, and returns a grounded answer with source
    citations (paper title + page number).
    """
    settings = get_settings()

    # ── Get or create conversation ──────────────────────────────────────────
    conversation = await get_or_create_conversation(db, request.conversation_id)

    # ── Fetch chat history for conversational memory ───────────────────────
    history = await get_chat_history(
        db,
        conversation.id,
        n_turns=settings.chat_history_turns,
    )

    # ── Save user message first ────────────────────────────────────────────
    await save_message(
        db,
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )

    # ── Run RAG pipeline ───────────────────────────────────────────────────
    try:
        result = await rag_query(
            db=db,
            question=request.message,
            retrieval_strategy=request.retrieval_strategy,
            embedding_model=request.embedding_model,
            chat_history=history,
            settings=settings,
        )
    except Exception as exc:
        logger.error("RAG pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline failed: {exc}",
        )

    # ── Save assistant response with sources ───────────────────────────────
    await save_message(
        db,
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
        sources=result["sources"],
        retrieval_strategy=result["retrieval_strategy"],
        embedding_model=result["embedding_model"],
    )

    # ── Return response ────────────────────────────────────────────────────
    sources = [SourceOut(**s) for s in result["sources"]]

    return ChatResponse(
        conversation_id=str(conversation.id),
        answer=result["answer"],
        sources=sources,
        retrieval_strategy=result["retrieval_strategy"],
        embedding_model=result["embedding_model"],
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
) -> ConversationOut:
    """Retrieve full message history for a conversation."""
    try:
        cid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid UUID format.")

    conversation = await get_or_create_conversation(db, cid)
    return ConversationOut.model_validate(conversation)
