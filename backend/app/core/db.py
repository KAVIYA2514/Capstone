"""
Async SQLAlchemy engine and session factory for Neon PostgreSQL.

Why asyncpg?
- asyncpg is the fastest Python PostgreSQL driver, and is required for
  SQLAlchemy's async extension. It speaks the raw Postgres wire protocol
  directly without going through DBAPI2.
- Neon's pooled connection string works natively with asyncpg.

Connection pool settings are conservative by default because Neon's free
tier limits concurrent connections; increase pool_size and max_overflow for
paid tiers or self-hosted Postgres.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


def clean_db_url(url: str) -> tuple[str, dict]:
    """Clean database URL to be compatible with asyncpg."""
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode  # noqa: PLC0415

    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    parsed = urlparse(url)
    query_params = dict(parse_qsl(parsed.query))

    connect_args = {}
    sslmode = query_params.get("sslmode", "").lower()
    if sslmode in ("require", "verify-ca", "verify-full") or "sslmode" not in query_params:
        connect_args["ssl"] = True

    # Strip asyncpg-unsupported query arguments
    asyncpg_unsupported = ["sslmode", "channel_binding"]
    cleaned_query = {k: v for k, v in query_params.items() if k not in asyncpg_unsupported}

    new_query_str = urlencode(cleaned_query)
    cleaned_parsed = parsed._replace(query=new_query_str)
    cleaned_url = urlunparse(cleaned_parsed)

    return cleaned_url, connect_args


def _build_engine() -> object:
    settings = get_settings()
    db_url, connect_args = clean_db_url(settings.database_url)

    return create_async_engine(
        db_url,
        connect_args=connect_args,
        echo=settings.log_level == "DEBUG",  # SQL logging in debug mode only
        pool_size=5,          # base connections kept alive
        max_overflow=10,      # extra connections allowed under load
        pool_pre_ping=False,   # disabled to prevent transaction state errors with Neon pooler
        pool_recycle=1800,    # recycle connections every 30 min (Neon idle timeout)
    )


# Module-level singletons — created once when the module is first imported.
engine = _build_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # objects remain usable after commit (important for async)
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a DB session per request.

    Usage:
        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
