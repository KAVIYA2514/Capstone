"""
Paper ORM model.

Represents a single ingested research paper PDF. One paper → many chunks.
The source_filename is used as the idempotency key in the seed script
(skip re-ingesting if a paper with the same filename already exists).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(  # noqa: F821
        "Chunk",
        back_populates="paper",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Paper id={self.id} title={self.title!r}>"
