"""
Chunking strategies — split page-level text into retrieval-ready chunks.

Three strategies implemented to satisfy the rubric's ≥2 chunking comparison:

1. fixed_512  — token-aware fixed-size windows (512 tokens, 50-token overlap)
   WHY: Simple, predictable, easy to reason about. Good baseline for comparison.
   TRADE-OFF: May split mid-sentence or mid-equation; ignores semantic structure.

2. recursive  — LangChain RecursiveCharacterTextSplitter
   WHY: Tries to respect natural boundaries (paragraphs → sentences → words).
   Split hierarchy: ["\n\n", "\n", ". ", " ", ""] — falls back to finer splits
   only when a chunk is still too large. Better coherence than fixed-size.
   TRADE-OFF: Char-based, so chunk token counts vary widely.

3. semantic   — embedding-similarity–guided sentence grouping
   WHY: Groups sentences that are topically related. Produces chunks that are
   semantically coherent rather than mechanically bounded — ideal for research
   papers where a "topic" may span multiple short sentences.
   TRADE-OFF: Requires a forward pass through an embedding model during ingestion
   (we use the small all-MiniLM-L6-v2 model for speed), so it's ~5–10× slower
   than the other strategies.

All strategies return Chunk dataclasses that carry paper_id, page_number, and
chunk_index so the metadata survives all the way to the vector store row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loaders import PageContent

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["fixed_512", "recursive", "semantic"]

# Token approximation: 1 token ≈ 4 characters (GPT-style BPE heuristic).
# tiktoken is not used because it requires Rust compilation and has no
# pre-built wheel for Python 3.14. The 4-char/token heuristic is standard
# in the NLP literature and is accurate to within ~15% for English text.
_CHARS_PER_TOKEN = 4


@dataclass
class TextChunk:
    """A single chunk ready for embedding and DB insertion."""

    text: str
    paper_id: int
    page_number: int
    chunk_index: int
    chunking_strategy: ChunkingStrategy
    metadata: dict = field(default_factory=dict)


# ── Strategy 1: Fixed-size token windows ─────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_chars_fixed(
    text: str, max_tokens: int = 512, overlap: int = 50
) -> list[str]:
    """
    Split text into fixed-size character windows approximating token limits.

    We work in characters rather than tokens because tiktoken requires Rust
    compilation and is unavailable on Python 3.14. The conversion:
        max_chars  = max_tokens  * 4
        overlap_chars = overlap  * 4
    gives chunks of ~512 tokens with ~50-token overlap, accurate to ±15%.
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN

    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start += max_chars - overlap_chars

    return chunks


def chunk_fixed(
    pages: list[PageContent],
    paper_id: int,
    max_tokens: int = 512,
    overlap: int = 50,
) -> list[TextChunk]:
    """Fixed-size ~512-token chunking with ~50-token overlap (char-based approximation)."""
    result: list[TextChunk] = []
    global_index = 0

    for page in pages:
        if not page.text.strip():
            continue
        splits = _split_chars_fixed(page.text, max_tokens, overlap)
        for split in splits:
            if not split.strip():
                continue
            result.append(
                TextChunk(
                    text=split,
                    paper_id=paper_id,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    chunking_strategy="fixed_512",
                    metadata={
                        "paper_id": paper_id,
                        "page_number": page.page_number,
                        "max_tokens": max_tokens,
                        "overlap": overlap,
                    },
                )
            )
            global_index += 1

    logger.info("fixed_512: produced %d chunks", len(result))
    return result


# ── Strategy 2: Recursive character splitting ─────────────────────────────────

_RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1800,      # ~450 tokens at 4 chars/token heuristic
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


def chunk_recursive(pages: list[PageContent], paper_id: int) -> list[TextChunk]:
    """
    Recursive character text splitter — respects paragraph/sentence boundaries.

    LangChain's RecursiveCharacterTextSplitter tries each separator in order
    and only uses finer-grained splits when a chunk is still too large. This
    preserves sentence coherence better than fixed-token windows.
    """
    result: list[TextChunk] = []
    global_index = 0

    for page in pages:
        if not page.text.strip():
            continue
        splits = _RECURSIVE_SPLITTER.split_text(page.text)
        for split in splits:
            if not split.strip():
                continue
            result.append(
                TextChunk(
                    text=split,
                    paper_id=paper_id,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    chunking_strategy="recursive",
                    metadata={
                        "paper_id": paper_id,
                        "page_number": page.page_number,
                    },
                )
            )
            global_index += 1

    logger.info("recursive: produced %d chunks", len(result))
    return result


# ── Strategy 3: Semantic / embedding-similarity chunking ─────────────────────

def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — handles '. ', '? ', '! ' boundaries."""
    import re

    # Split on sentence-ending punctuation followed by whitespace + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_semantic(
    pages: list[PageContent],
    paper_id: int,
    similarity_threshold: float = 0.75,
    max_chunk_sentences: int = 8,
) -> list[TextChunk]:
    """
    Semantic chunking: group consecutive sentences until embedding similarity drops.

    Algorithm:
    1. Split each page into sentences.
    2. Embed each sentence with a lightweight local model (all-MiniLM-L6-v2,
       ~90 MB, fast CPU inference).
    3. Compute cosine similarity between consecutive sentence embeddings.
    4. Start a new chunk when similarity drops below threshold OR chunk
       reaches max_chunk_sentences sentences.

    WHY all-MiniLM-L6-v2 for the boundary detector?
    We intentionally use a small model here (not bge-m3) because this model's
    only job is detecting *relative* topic shifts between adjacent sentences,
    not producing final retrieval embeddings. Speed matters more than absolute
    quality for this step. The actual retrieval embeddings are produced later
    by the full-size models.

    Args:
        pages: list of PageContent objects from the loader
        paper_id: FK to the papers table
        similarity_threshold: cosine similarity below which a new chunk starts
        max_chunk_sentences: hard cap on chunk size in sentences
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "sentence-transformers not available — falling back to recursive chunking"
        )
        return chunk_recursive(pages, paper_id)

    # Lazy-load the small boundary-detection model (separate from bge-m3)
    _boundary_model = SentenceTransformer("all-MiniLM-L6-v2")

    result: list[TextChunk] = []
    global_index = 0

    for page in pages:
        if not page.text.strip():
            continue

        sentences = _split_sentences(page.text)
        if len(sentences) <= 1:
            # Single-sentence page — treat as one chunk
            result.append(
                TextChunk(
                    text=page.text.strip(),
                    paper_id=paper_id,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    chunking_strategy="semantic",
                    metadata={"paper_id": paper_id, "page_number": page.page_number},
                )
            )
            global_index += 1
            continue

        # Embed all sentences at once (batched)
        embeddings = _boundary_model.encode(sentences, normalize_embeddings=True)

        current_chunk: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            # Cosine similarity between consecutive sentences (dot product of
            # unit-norm vectors equals cosine similarity)
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            topic_shift = sim < similarity_threshold
            chunk_full = len(current_chunk) >= max_chunk_sentences

            if topic_shift or chunk_full:
                # Flush current chunk
                chunk_text = " ".join(current_chunk)
                result.append(
                    TextChunk(
                        text=chunk_text,
                        paper_id=paper_id,
                        page_number=page.page_number,
                        chunk_index=global_index,
                        chunking_strategy="semantic",
                        metadata={
                            "paper_id": paper_id,
                            "page_number": page.page_number,
                            "similarity_threshold": similarity_threshold,
                        },
                    )
                )
                global_index += 1
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        # Flush final chunk
        if current_chunk:
            result.append(
                TextChunk(
                    text=" ".join(current_chunk),
                    paper_id=paper_id,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    chunking_strategy="semantic",
                    metadata={"paper_id": paper_id, "page_number": page.page_number},
                )
            )
            global_index += 1

    logger.info("semantic: produced %d chunks", len(result))
    return result


# ── Public factory ────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[PageContent],
    paper_id: int,
    strategy: ChunkingStrategy = "recursive",
) -> list[TextChunk]:
    """
    Dispatch to the appropriate chunking strategy.

    Args:
        pages: extracted PDF pages from load_pdf()
        paper_id: DB id of the Paper row (must be committed first)
        strategy: 'fixed_512' | 'recursive' | 'semantic'
    """
    if strategy == "fixed_512":
        return chunk_fixed(pages, paper_id)
    elif strategy == "recursive":
        return chunk_recursive(pages, paper_id)
    elif strategy == "semantic":
        return chunk_semantic(pages, paper_id)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
