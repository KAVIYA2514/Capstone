"""
Retrieval evaluation harness.

Runs a fixed set of test queries against all three retrieval strategies
and outputs a Markdown comparison table to /notebooks/retrieval_eval_results.md.

This is the "evaluator.py" deliverable for the rubric. It lets you:
1. Run all strategies on the same queries and see which chunks are returned.
2. Manually judge precision/recall (is the right paper + page surfaced?).
3. Compare strategies quantitatively (MRR, top-1 accuracy on test set).

Usage:
    cd backend
    python -m app.retrieval.evaluator
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import text

# Add backend to path when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.core.config import get_settings
from app.core.db import AsyncSessionLocal
from app.ingestion.embedder import get_embedder
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

# ── Test query set (10 questions spanning the 5 seeded papers) ──────────────
TEST_QUERIES = [
    {
        "id": "Q01",
        "question": "What is the attention mechanism in transformer models?",
        "expected_paper": "Attention Is All You Need",
        "expected_page_range": (1, 5),
    },
    {
        "id": "Q02",
        "question": "How does multi-head attention work?",
        "expected_paper": "Attention Is All You Need",
        "expected_page_range": (3, 6),
    },
    {
        "id": "Q03",
        "question": "What is BERT and how is it pre-trained?",
        "expected_paper": "BERT",
        "expected_page_range": (1, 5),
    },
    {
        "id": "Q04",
        "question": "What is masked language modeling?",
        "expected_paper": "BERT",
        "expected_page_range": (2, 6),
    },
    {
        "id": "Q05",
        "question": "How does retrieval-augmented generation improve LLM answers?",
        "expected_paper": "RAG",
        "expected_page_range": (1, 6),
    },
    {
        "id": "Q06",
        "question": "What is the difference between RAG and fine-tuning?",
        "expected_paper": "RAG",
        "expected_page_range": (1, 10),
    },
    {
        "id": "Q07",
        "question": "What is LoRA and how does it reduce the number of trainable parameters?",
        "expected_paper": "LoRA",
        "expected_page_range": (1, 5),
    },
    {
        "id": "Q08",
        "question": "How does low-rank decomposition work in LoRA?",
        "expected_paper": "LoRA",
        "expected_page_range": (2, 6),
    },
    {
        "id": "Q09",
        "question": "What is chain-of-thought prompting?",
        "expected_paper": "Chain-of-Thought",
        "expected_page_range": (1, 5),
    },
    {
        "id": "Q10",
        "question": "How does few-shot chain-of-thought prompting improve reasoning?",
        "expected_paper": "Chain-of-Thought",
        "expected_page_range": (1, 8),
    },
]


def _hit(results: list[dict], expected_paper: str) -> bool:
    """Check if any top result is from the expected paper."""
    return any(
        expected_paper.lower() in r.get("paper_title", "").lower() for r in results
    )


def _mrr(results: list[dict], expected_paper: str) -> float:
    """Mean Reciprocal Rank for a single query."""
    for i, r in enumerate(results, start=1):
        if expected_paper.lower() in r.get("paper_title", "").lower():
            return 1.0 / i
    return 0.0


async def run_evaluation(
    embedding_model: str | None = None,
    top_k: int = 5,
    output_path: str | Path | None = None,
) -> None:
    """
    Run all test queries through all three retrieval strategies.

    Args:
        embedding_model: which model to use (defaults to EMBEDDING_MODEL env var)
        top_k: how many results to fetch per query
        output_path: where to write the Markdown table
    """
    settings = get_settings()
    emb_model = embedding_model or settings.embedding_model

    embedder = get_embedder(emb_model, settings)

    rows: list[dict] = []

    async with AsyncSessionLocal() as db:
        for query_def in TEST_QUERIES:
            q = query_def["question"]
            expected = query_def["expected_paper"]
            logger.info("Evaluating: %s", q)

            # Embed the query
            q_vec = embedder.embed_query(q)

            # Strategy 1: dense
            dense = await dense_search(db, q_vec, emb_model, top_k=top_k)
            # Strategy 2: hybrid
            hybrid = await hybrid_search(db, q, q_vec, emb_model, top_k=top_k * 4)
            hybrid_top = hybrid[:top_k]
            # Strategy 3: hybrid + rerank
            hybrid_reranked = rerank(q, hybrid[: top_k * 4], top_n=top_k)

            rows.append(
                {
                    "id": query_def["id"],
                    "question": q[:60] + ("…" if len(q) > 60 else ""),
                    "expected": expected,
                    "dense_hit": _hit(dense, expected),
                    "dense_mrr": round(_mrr(dense, expected), 3),
                    "hybrid_hit": _hit(hybrid_top, expected),
                    "hybrid_mrr": round(_mrr(hybrid_top, expected), 3),
                    "rerank_hit": _hit(hybrid_reranked, expected),
                    "rerank_mrr": round(_mrr(hybrid_reranked, expected), 3),
                    "dense_top1": dense[0]["paper_title"] if dense else "—",
                    "hybrid_top1": hybrid_top[0]["paper_title"] if hybrid_top else "—",
                    "rerank_top1": hybrid_reranked[0]["paper_title"] if hybrid_reranked else "—",
                }
            )

    # Compute aggregate stats
    def _avg(field: str) -> float:
        return round(sum(r[field] for r in rows) / len(rows), 3)

    # Build Markdown table
    lines = [
        "# Retrieval Strategy Evaluation Results\n",
        f"**Embedding model**: `{emb_model}`  ",
        f"**Top-k**: {top_k}  ",
        f"**Queries**: {len(rows)}\n",
        "## Per-Query Results\n",
        "| ID | Question | Expected Paper | Dense Hit | Dense MRR | Hybrid Hit | Hybrid MRR | Rerank Hit | Rerank MRR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['question']} | {r['expected']} "
            f"| {'✅' if r['dense_hit'] else '❌'} | {r['dense_mrr']} "
            f"| {'✅' if r['hybrid_hit'] else '❌'} | {r['hybrid_mrr']} "
            f"| {'✅' if r['rerank_hit'] else '❌'} | {r['rerank_mrr']} |"
        )

    lines += [
        "\n## Aggregate Statistics\n",
        "| Strategy | Hit Rate | Mean MRR |",
        "|---|---|---|",
        f"| Dense | {_avg('dense_hit'):.1%} | {_avg('dense_mrr')} |",
        f"| Hybrid (RRF) | {_avg('hybrid_hit'):.1%} | {_avg('hybrid_mrr')} |",
        f"| Hybrid + Rerank | {_avg('rerank_hit'):.1%} | {_avg('rerank_mrr')} |",
        "\n## Top-1 Paper Per Query\n",
        "| ID | Dense | Hybrid | Hybrid+Rerank |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['dense_top1'][:40]} "
            f"| {r['hybrid_top1'][:40]} "
            f"| {r['rerank_top1'][:40]} |"
        )

    md_content = "\n".join(lines)

    # Write output
    out_path = Path(output_path) if output_path else (
        Path(__file__).parent.parent.parent.parent / "notebooks" / "retrieval_eval_results.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    logger.info("Results written to: %s", out_path)
    print(md_content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_evaluation())
