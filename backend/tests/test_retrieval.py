"""
Unit tests for retrieval modules.

These tests mock the database to avoid requiring a live Neon connection.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── RRF fusion tests ──────────────────────────────────────────────────────────

class TestRRFFusion:
    """Test the Reciprocal Rank Fusion algorithm directly."""

    def test_basic_fusion(self):
        from app.retrieval.hybrid import _reciprocal_rank_fusion

        list1 = [
            {"chunk_id": 1, "paper_title": "Paper A", "page_number": 1, "chunk_text": "...", "score": 0.9},
            {"chunk_id": 2, "paper_title": "Paper B", "page_number": 2, "chunk_text": "...", "score": 0.8},
        ]
        list2 = [
            {"chunk_id": 2, "paper_title": "Paper B", "page_number": 2, "chunk_text": "...", "score": 0.95},
            {"chunk_id": 3, "paper_title": "Paper C", "page_number": 3, "chunk_text": "...", "score": 0.7},
        ]

        fused = _reciprocal_rank_fusion([list1, list2])

        # chunk_id=2 appears in both lists → should have higher RRF score
        assert fused[0]["chunk_id"] == 2, "Chunk appearing in both lists should rank first"
        assert all("rrf_score" in r for r in fused), "All results should have rrf_score"

    def test_empty_lists(self):
        from app.retrieval.hybrid import _reciprocal_rank_fusion

        assert _reciprocal_rank_fusion([[], []]) == []

    def test_single_list(self):
        from app.retrieval.hybrid import _reciprocal_rank_fusion

        items = [
            {"chunk_id": i, "paper_title": "P", "page_number": i, "chunk_text": "t", "score": 1.0}
            for i in range(5)
        ]
        fused = _reciprocal_rank_fusion([items])
        assert len(fused) == 5

    def test_scores_are_positive(self):
        from app.retrieval.hybrid import _reciprocal_rank_fusion

        items = [
            {"chunk_id": i, "paper_title": "P", "page_number": 1, "chunk_text": "t", "score": 0.5}
            for i in range(10)
        ]
        fused = _reciprocal_rank_fusion([items])
        assert all(r["rrf_score"] > 0 for r in fused)

    def test_descending_order(self):
        from app.retrieval.hybrid import _reciprocal_rank_fusion

        items = [
            {"chunk_id": i, "paper_title": "P", "page_number": i, "chunk_text": "t", "score": 0.5}
            for i in range(5)
        ]
        fused = _reciprocal_rank_fusion([items])
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True), "Results should be in descending score order"


# ── Reranker tests ────────────────────────────────────────────────────────────

class TestReranker:
    def test_empty_candidates(self):
        from app.retrieval.reranker import rerank

        assert rerank("query", []) == []

    def test_top_n_limit(self):
        """rerank should return at most top_n results."""
        from app.retrieval.reranker import rerank

        candidates = [
            {"chunk_id": i, "paper_title": "P", "page_number": 1,
             "chunk_text": f"Text chunk number {i}", "score": 0.5}
            for i in range(10)
        ]

        # Mock the CrossEncoder to avoid downloading the model in tests
        mock_scores = list(range(10, 0, -1))  # 10, 9, 8, ..., 1
        with patch("app.retrieval.reranker._get_crossencoder") as mock_get:
            mock_ce = MagicMock()
            mock_ce.predict.return_value = mock_scores
            mock_get.return_value = mock_ce

            results = rerank("test query", candidates, top_n=3)

        assert len(results) == 3

    def test_original_rank_added(self):
        from app.retrieval.reranker import rerank

        candidates = [
            {"chunk_id": i, "paper_title": "P", "page_number": 1,
             "chunk_text": f"Chunk {i}", "score": 0.5}
            for i in range(3)
        ]

        mock_scores = [0.8, 0.6, 0.9]
        with patch("app.retrieval.reranker._get_crossencoder") as mock_get:
            mock_ce = MagicMock()
            mock_ce.predict.return_value = mock_scores
            mock_get.return_value = mock_ce

            results = rerank("query", candidates, top_n=3)

        assert all("original_rank" in r for r in results)
