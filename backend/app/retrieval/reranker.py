"""
Cross-encoder reranking on top of hybrid retrieval candidates.

WHY reranking?
Bi-encoder retrieval (dense/hybrid) is fast because both query and document
are encoded independently — but this means the model can't attend to the
interaction between query and document tokens directly.

A cross-encoder (a.k.a. "reader" or "reranker") takes the query and each
candidate document *together* as input, allowing full cross-attention over
both. This is much more expensive (O(k) forward passes, one per candidate),
but produces far superior relevance scores.

Strategy:
- Use hybrid search to fetch top-20 candidates quickly (efficient recall)
- Re-score each candidate with the cross-encoder (expensive but high precision)
- Return the top-5 re-ranked results

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on the MS MARCO passage retrieval dataset (QA-style)
- 6-layer MiniLM → small and fast (CPU-viable)
- Achieves strong MRR@10 on TREC DL despite its small size
- The model is downloaded on first use (~68 MB) and cached locally

Trade-off:
Reranking adds ~1–3 seconds latency per query on CPU (depending on
candidate count and text length). For the demo we expose this as an
optional strategy so users can compare quality vs. latency directly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Cross-encoder model — lazy-loaded on first call to avoid blocking startup
_CROSSENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_crossencoder = None


def _get_crossencoder() -> object:
    """Lazy-load the CrossEncoder model (downloads ~68 MB on first call)."""
    global _crossencoder
    if _crossencoder is None:
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info("Loading CrossEncoder model: %s", _CROSSENCODER_MODEL_NAME)
            _crossencoder = CrossEncoder(_CROSSENCODER_MODEL_NAME)
            logger.info("CrossEncoder loaded.")
        except Exception as exc:
            logger.warning(
                "Could not load CrossEncoder model (error: %s). "
                "Using fallback query-term matching for reranking.",
                exc
            )
            _crossencoder = "fallback"
    return _crossencoder


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Re-score retrieval candidates with a cross-encoder and return the top-n.

    Args:
        query: the original user query string
        candidates: list of chunk dicts from hybrid_search()
                    (each must have 'chunk_text')
        top_n: number of results to return after reranking

    Returns:
        Top-n candidates sorted by cross-encoder score (descending),
        with 'score' updated to the cross-encoder logit value and
        'original_rank' added for comparison.
    """
    if not candidates:
        return []

    crossencoder = _get_crossencoder()

    if crossencoder == "fallback":
        # Fallback ranking logic: score based on simple term overlaps
        query_words = set(query.lower().split())
        for i, candidate in enumerate(candidates):
            candidate["original_rank"] = i + 1
            text_words = set(candidate["chunk_text"].lower().split())
            overlap = len(query_words.intersection(text_words))
            # Boost the existing score with word overlap count
            candidate["score"] = float(candidate["score"] + overlap * 0.1)

        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        logger.debug("rerank: fallback word-overlap scoring completed.")
        return reranked[:top_n]

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, c["chunk_text"]) for c in candidates]

    # Predict relevance scores — CrossEncoder returns logits (not probabilities),
    # but ranking order is preserved. Higher = more relevant.
    scores = crossencoder.predict(pairs)

    # Attach scores and sort
    for i, (candidate, score) in enumerate(zip(candidates, scores)):
        candidate["original_rank"] = i + 1
        candidate["score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

    logger.debug(
        "rerank: %d candidates → top %d. "
        "Score range: [%.3f, %.3f]",
        len(candidates),
        top_n,
        min(scores),
        max(scores),
    )

    return reranked[:top_n]

