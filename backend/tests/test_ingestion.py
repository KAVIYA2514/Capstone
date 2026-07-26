"""
Unit tests for the ingestion pipeline (chunking strategies + embedder interface).
"""

from __future__ import annotations

import pytest

from app.ingestion.chunking import (
    TextChunk,
    chunk_fixed,
    chunk_recursive,
    chunk_pages,
)
from app.ingestion.loaders import PageContent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pages() -> list[PageContent]:
    """Three pages of synthetic text for testing chunkers."""
    return [
        PageContent(
            page_number=1,
            text=(
                "The Transformer architecture introduced the attention mechanism. "
                "Self-attention allows the model to weigh the importance of each token "
                "relative to all other tokens in the sequence. "
                "Multi-head attention runs this process in parallel across multiple heads, "
                "each learning different aspects of the input relationships. "
                "This design eliminates the need for recurrence, making training highly parallelisable."
            ),
        ),
        PageContent(
            page_number=2,
            text=(
                "BERT pre-trains deep bidirectional representations by jointly conditioning "
                "on both left and right context in all layers. "
                "The pre-training uses two tasks: Masked Language Modeling and Next Sentence Prediction. "
                "In MLM, 15% of input tokens are replaced with a [MASK] token and the model "
                "must predict the original token."
            ),
        ),
        PageContent(
            page_number=3,
            text=(
                "LoRA decomposes weight updates into two low-rank matrices A and B "
                "such that ΔW = BA, where r << min(d,k). "
                "Only A and B are trained; the original weights remain frozen. "
                "This reduces trainable parameters from d×k to r×(d+k)."
            ),
        ),
    ]


# ── Fixed-512 chunker tests ───────────────────────────────────────────────────

class TestFixedChunker:
    def test_produces_chunks(self, sample_pages):
        chunks = chunk_fixed(sample_pages, paper_id=1)
        assert len(chunks) > 0, "Should produce at least one chunk"

    def test_chunk_type(self, sample_pages):
        chunks = chunk_fixed(sample_pages, paper_id=1)
        for c in chunks:
            assert isinstance(c, TextChunk)

    def test_chunk_strategy_tag(self, sample_pages):
        chunks = chunk_fixed(sample_pages, paper_id=1)
        for c in chunks:
            assert c.chunking_strategy == "fixed_512"

    def test_chunk_paper_id(self, sample_pages):
        chunks = chunk_fixed(sample_pages, paper_id=42)
        for c in chunks:
            assert c.paper_id == 42

    def test_page_numbers_preserved(self, sample_pages):
        chunks = chunk_fixed(sample_pages, paper_id=1)
        page_nums = {c.page_number for c in chunks}
        # Should have chunks from all 3 pages (since text fits in one chunk each)
        assert page_nums.issubset({1, 2, 3})
        assert len(page_nums) >= 1

    def test_no_empty_chunks(self, sample_pages):
        chunks = chunk_fixed(sample_pages, paper_id=1)
        for c in chunks:
            assert c.text.strip(), "Chunks should not be empty"

    def test_skips_empty_pages(self):
        pages = [
            PageContent(page_number=1, text="   "),
            PageContent(page_number=2, text="Real content here."),
        ]
        chunks = chunk_fixed(pages, paper_id=1)
        assert all(c.text.strip() for c in chunks)


# ── Recursive chunker tests ───────────────────────────────────────────────────

class TestRecursiveChunker:
    def test_produces_chunks(self, sample_pages):
        chunks = chunk_recursive(sample_pages, paper_id=1)
        assert len(chunks) > 0

    def test_strategy_tag(self, sample_pages):
        chunks = chunk_recursive(sample_pages, paper_id=1)
        for c in chunks:
            assert c.chunking_strategy == "recursive"

    def test_metadata_has_page_number(self, sample_pages):
        chunks = chunk_recursive(sample_pages, paper_id=1)
        for c in chunks:
            assert "page_number" in c.metadata

    def test_chunk_indices_sequential(self, sample_pages):
        chunks = chunk_recursive(sample_pages, paper_id=1)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# ── Factory function tests ────────────────────────────────────────────────────

class TestChunkPages:
    def test_dispatches_fixed(self, sample_pages):
        chunks = chunk_pages(sample_pages, paper_id=1, strategy="fixed_512")
        assert all(c.chunking_strategy == "fixed_512" for c in chunks)

    def test_dispatches_recursive(self, sample_pages):
        chunks = chunk_pages(sample_pages, paper_id=1, strategy="recursive")
        assert all(c.chunking_strategy == "recursive" for c in chunks)

    def test_unknown_strategy_raises(self, sample_pages):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_pages(sample_pages, paper_id=1, strategy="invalid")  # type: ignore


# ── Embedder interface tests ───────────────────────────────────────────────────

class TestEmbedderInterface:
    """Test the EmbedderBase interface and factory without calling external APIs."""

    def test_get_embedder_nvidia(self, monkeypatch):
        """get_embedder('nvidia/...') should return a NvidiaEmbedder."""
        from app.ingestion.embedder import get_embedder, NvidiaEmbedder

        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.setenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")

        # Clear settings cache
        from app.core.config import get_settings
        get_settings.cache_clear()

        embedder = get_embedder("nvidia/nv-embedqa-e5-v5")
        assert isinstance(embedder, NvidiaEmbedder)

    def test_get_embedder_bge(self):
        """get_embedder('BAAI/bge-m3') should return a BGEEmbedder."""
        from app.ingestion.embedder import get_embedder, BGEEmbedder

        embedder = get_embedder("BAAI/bge-m3")
        assert isinstance(embedder, BGEEmbedder)

    def test_unknown_model_raises(self):
        from app.ingestion.embedder import get_embedder

        with pytest.raises(ValueError, match="Unknown embedding model"):
            get_embedder("some/unknown-model")
