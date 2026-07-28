"""
Application configuration — all values loaded from environment variables.

Design rationale:
- pydantic-settings gives us type-safe, validated env-var loading with zero
  boilerplate. Every setting has a type, an optional default, and can be
  overridden at runtime by the environment (Docker, CI, or a local .env file).
- Using @lru_cache on get_settings() means the Settings object is constructed
  once per process lifetime, not on every request — important for
  sentence-transformers which loads a heavyweight model at class instantiation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── Database ───────────────────────────────────────────────────────────
    # Neon connection string — use the pooled endpoint (port 5432).
    # asyncpg driver is required for async SQLAlchemy.
    # Example: postgresql+asyncpg://user:pass@ep-xxx.neon.tech/dbname?sslmode=require
    database_url: str

    # ── NVIDIA NIM ─────────────────────────────────────────────────────────
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # ── Embedding ──────────────────────────────────────────────────────────
    # Primary: NVIDIA-hosted (no local GPU required)
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embedding_dim: int = 1024  # output dimensionality for the primary model

    # Secondary: local sentence-transformers (used for comparison / offline dev)
    # BAAI/bge-m3 is also 1024-dim, so both fit in the same vector(1024) column.
    local_embedding_model: str = "BAAI/bge-m3"

    # ── LLM ────────────────────────────────────────────────────────────────
    llm_provider: Literal["nvidia_nim", "ollama"] = "nvidia_nim"
    chat_model: str = "meta/llama-3.1-8b-instruct"

    # Ollama fallback — only used when llm_provider = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # ── App ────────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Retrieval ──────────────────────────────────────────────────────────
    default_retrieval_strategy: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid"
    default_top_k: int = 20      # candidates fetched before reranking
    reranker_top_n: int = 5      # final results after cross-encoder rerank

    # ── Conversational Memory ──────────────────────────────────────────────
    chat_history_turns: int = 5  # last N user+assistant message pairs

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Allow CORS_ORIGINS as a comma-separated string in the .env file."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
