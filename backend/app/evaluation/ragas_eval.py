"""
RAGAS evaluation script — scores the RAG pipeline on 10 test questions.

Uses RAGAS metrics:
- faithfulness:        Does the answer stay grounded in the retrieved context?
- answer_relevancy:    Is the answer relevant to the question?
- context_precision:   Are the retrieved chunks actually relevant to the question?
- context_recall:      Are all necessary chunks retrieved? (requires ground truth)

We use LLM-as-judge mode (RAGAS internally calls the LLM to score answers),
so NVIDIA_API_KEY must be set. RAGAS supports OpenAI-compatible endpoints via
the langchain_openai integration.

Usage:
    cd backend
    python -m app.evaluation.ragas_eval

Output: prints a markdown table of scores + writes results.md to /notebooks/
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# ── 10 test questions with reference answers for RAGAS ────────────────────────
EVAL_DATASET = [
    {
        "question": "What is the attention mechanism in transformer models?",
        "ground_truth": "The attention mechanism allows the model to weigh the importance of different input positions. In transformers, multi-head attention computes scaled dot-product attention in parallel across multiple heads.",
    },
    {
        "question": "How does multi-head attention differ from single-head attention?",
        "ground_truth": "Multi-head attention runs the attention function in parallel multiple times with different learned linear projections, then concatenates the results. This allows the model to jointly attend to information from different representation subspaces.",
    },
    {
        "question": "What pre-training objectives does BERT use?",
        "ground_truth": "BERT uses two pre-training objectives: Masked Language Modeling (MLM), where 15% of tokens are masked and the model predicts them, and Next Sentence Prediction (NSP), where the model predicts whether two sentences are consecutive.",
    },
    {
        "question": "How does retrieval-augmented generation work?",
        "ground_truth": "RAG combines a retrieval component (a dense retriever like DPR) with a sequence-to-sequence generator. For each input, it retrieves relevant documents from a knowledge base and conditions the generator on both the input and retrieved documents.",
    },
    {
        "question": "What problem does LoRA solve for large language model fine-tuning?",
        "ground_truth": "LoRA addresses the prohibitive cost of full fine-tuning of large models by injecting trainable low-rank decomposition matrices into existing weights. This reduces the number of trainable parameters by orders of magnitude while maintaining model quality.",
    },
    {
        "question": "What is chain-of-thought prompting?",
        "ground_truth": "Chain-of-thought prompting is a technique where a series of intermediate reasoning steps are included in few-shot examples, eliciting the model to produce a chain of reasoning before giving the final answer.",
    },
    {
        "question": "What is the role of the encoder and decoder in the Transformer?",
        "ground_truth": "The encoder maps the input sequence to a sequence of continuous representations. The decoder generates the output sequence auto-regressively, attending to both the encoder output (via cross-attention) and previously generated tokens (via masked self-attention).",
    },
    {
        "question": "How does LoRA apply low-rank decomposition to model weights?",
        "ground_truth": "LoRA represents weight updates as the product of two low-rank matrices: ΔW = BA, where B is d×r and A is r×k, with r << min(d,k). Only A and B are trained; the original weights W are frozen.",
    },
    {
        "question": "What dataset was used to evaluate chain-of-thought prompting?",
        "ground_truth": "Chain-of-thought prompting was evaluated on arithmetic (GSM8K, SVAMP, ASDiv), commonsense (StrategyQA, ARC), and symbolic reasoning benchmarks, showing significant improvements over standard prompting on larger models.",
    },
    {
        "question": "What is positional encoding in the Transformer architecture?",
        "ground_truth": "Positional encoding adds information about the position of each token in the sequence to its embedding. The original Transformer uses fixed sinusoidal positional encodings with different frequencies for each dimension.",
    },
]


async def run_ragas_evaluation(
    retrieval_strategy: str = "hybrid",
    embedding_model: str | None = None,
    output_path: str | Path | None = None,
) -> None:
    """
    Run RAGAS evaluation on the test dataset.

    This function:
    1. Runs each test question through the full RAG pipeline
    2. Collects (question, answer, contexts, ground_truth) for each
    3. Scores them with RAGAS metrics
    4. Outputs a markdown results table
    """
    try:
        from ragas import evaluate  # noqa: PLC0415
        from ragas.metrics import (  # noqa: PLC0415
            faithfulness,
            answer_relevancy,
            context_precision,
        )
        from datasets import Dataset  # noqa: PLC0415
    except ImportError:
        logger.error(
            "RAGAS not installed. Run: pip install ragas datasets"
        )
        return

    from app.core.config import get_settings  # noqa: PLC0415
    from app.core.db import AsyncSessionLocal  # noqa: PLC0415
    from app.ingestion.embedder import get_embedder  # noqa: PLC0415
    from app.retrieval.hybrid import hybrid_search  # noqa: PLC0415
    from app.retrieval.dense import dense_search  # noqa: PLC0415
    from app.rag.chain import rag_query  # noqa: PLC0415

    settings = get_settings()
    emb_model = embedding_model or settings.embedding_model
    embedder = get_embedder(emb_model, settings)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    logger.info("Running RAG pipeline on %d test questions...", len(EVAL_DATASET))

    async with AsyncSessionLocal() as db:
        for item in EVAL_DATASET:
            q = item["question"]
            logger.info("  Q: %s", q[:60])

            try:
                result = await rag_query(
                    db=db,
                    question=q,
                    retrieval_strategy=retrieval_strategy,
                    embedding_model=emb_model,
                )

                chunk_texts = [s["chunk_text"] for s in result["sources"]]

                questions.append(q)
                answers.append(result["answer"])
                contexts.append(chunk_texts)
                ground_truths.append(item["ground_truth"])
                logger.info("    ✓ Got answer (%d chars)", len(result["answer"]))

            except Exception as exc:
                logger.warning("    ✗ Failed: %s", exc)
                questions.append(q)
                answers.append("Error: " + str(exc))
                contexts.append([""])
                ground_truths.append(item["ground_truth"])

    # Build RAGAS dataset
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    logger.info("Scoring with RAGAS...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        df = result.to_pandas()

        # Write markdown output
        md_lines = [
            "# RAGAS Evaluation Results\n",
            f"**Retrieval strategy**: `{retrieval_strategy}`  ",
            f"**Embedding model**: `{emb_model}`  ",
            f"**Questions evaluated**: {len(questions)}\n",
            "## Per-Question Scores\n",
        ]
        md_lines.append(df.to_markdown(index=False))

        agg = df[["faithfulness", "answer_relevancy", "context_precision"]].mean()
        md_lines += [
            "\n## Aggregate Scores\n",
            "| Metric | Score |",
            "|---|---|",
            f"| Faithfulness | {agg.get('faithfulness', 0):.3f} |",
            f"| Answer Relevancy | {agg.get('answer_relevancy', 0):.3f} |",
            f"| Context Precision | {agg.get('context_precision', 0):.3f} |",
        ]

        md_content = "\n".join(md_lines)
        out_path = Path(output_path) if output_path else (
            Path(__file__).parent.parent.parent.parent / "notebooks" / "ragas_results.md"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_content, encoding="utf-8")
        logger.info("Results written to: %s", out_path)
        print(md_content)

    except Exception as exc:
        logger.error("RAGAS scoring failed: %s", exc)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_ragas_evaluation())
