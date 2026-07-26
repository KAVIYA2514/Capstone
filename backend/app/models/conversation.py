"""
Conversation and Message ORM models.

Implements stretch goal 1: conversational memory stored in Postgres.

Design:
- Conversation is a lightweight session container (just a UUID + timestamp).
  The UUID primary key is generated server-side so there's no integer ID to
  guess / enumerate.
- Message stores the full exchange: role, content, sources (the chunk IDs +
  scores returned by retrieval), and which retrieval strategy was used.
  The sources JSONB field is what populates the SourceCitations UI component.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id}>"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # 'user' | 'assistant'
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Serialised retrieval result for this message.
    # Schema: [{paper_title, page_number, chunk_id, score, chunk_text?}, ...]
    # Stored as JSONB so the SourceCitations component can render it directly
    # without a round-trip to the chunks table.
    sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)

    # Auditing: which retrieval strategy / embedding model was active for this turn
    retrieval_strategy: Mapped[str | None] = mapped_column(String(50), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role!r} conv={self.conversation_id}>"
