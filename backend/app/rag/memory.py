"""
Conversational memory — fetch and format chat history from the DB.

WHY persist memory in PostgreSQL?
Rather than keeping a server-side in-memory session store (which resets
on restart and doesn't scale across multiple server instances), we store
every message in the `messages` table. This gives us:
- Persistent history across server restarts
- Queryable audit trail (what strategy was used for which answer?)
- The sources JSONB field links each assistant turn to the specific chunks
  that grounded it — useful for debugging retrieval quality

Memory window:
We fetch the last N turns (user + assistant pairs), controlled by the
CHAT_HISTORY_TURNS setting. Too many turns → expensive prompt; too few →
assistant "forgets" context. 5 turns (=10 messages) is a sensible default.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation, Message

logger = logging.getLogger(__name__)


async def get_or_create_conversation(
    db: AsyncSession,
    conversation_id: str | uuid.UUID | None = None,
) -> Conversation:
    """
    Return an existing Conversation or create a new one.

    Args:
        db: async session
        conversation_id: UUID string or UUID object (optional)

    Returns:
        Conversation ORM object (committed to DB)
    """
    if conversation_id:
        cid = uuid.UUID(str(conversation_id))
        result = await db.execute(
            select(Conversation).where(Conversation.id == cid)
        )
        conv = result.scalar_one_or_none()
        if conv:
            return conv
        # If ID was provided but not found, create with that ID
        conv = Conversation(id=cid)
    else:
        conv = Conversation()

    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def get_chat_history(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    n_turns: int = 5,
) -> list[dict]:
    """
    Fetch the last n_turns of chat history for a conversation.

    Returns a list of {role, content} dicts ordered chronologically
    (oldest first), ready to be passed to the RAG chain as chat history.

    Each "turn" is one user message + one assistant message = 2 rows,
    so we fetch 2 * n_turns messages.
    """
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(n_turns * 2)
    )
    messages = result.scalars().all()
    # Reverse to chronological order
    messages = list(reversed(messages))

    return [{"role": msg.role, "content": msg.content} for msg in messages]


async def save_message(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    sources: list[dict] | None = None,
    retrieval_strategy: str | None = None,
    embedding_model: str | None = None,
) -> Message:
    """
    Persist a single message to the messages table.

    Args:
        db: async session
        conversation_id: UUID of the parent conversation
        role: 'user' | 'assistant'
        content: message text
        sources: list of source dicts [{paper_title, page_number, score, ...}]
        retrieval_strategy: which strategy was used (for assistant turns)
        embedding_model: which embedding model was used (for assistant turns)

    Returns:
        The committed Message ORM object
    """
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
        retrieval_strategy=retrieval_strategy,
        embedding_model=embedding_model,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    logger.debug("Saved message id=%s role=%s conv=%s", msg.id, role, conversation_id)
    return msg
