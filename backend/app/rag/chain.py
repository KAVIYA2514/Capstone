"""
RAG chain — LCEL-based retrieval-augmented generation pipeline.

Architecture (LCEL):
  retrieve → format_context → build_messages → LLM → parse_output

LLM Provider switching:
- LLM_PROVIDER=nvidia_nim → ChatOpenAI pointed at NVIDIA NIM base URL
- LLM_PROVIDER=ollama    → ChatOpenAI pointed at local Ollama server
Both use the OpenAI-compatible API, so LangChain's ChatOpenAI class works
unchanged for both. This is the "generation trade-off" talking point:
  - NVIDIA NIM (meta/llama-4-maverick-17b-128e-instruct): hosted 17B MoE
    model, high quality, requires API key, ~1-3 s/response
  - Ollama (llama3.1:8b): local 8B model, ~3-8 s/response on CPU,
    fully offline, free after pull

Prompt design:
The system prompt enforces strict groundedness:
1. "Answer ONLY from the provided context" → reduces hallucination
2. "If context is insufficient, say 'I don't know'" → honest fallback
3. Sources are quoted verbatim in the prompt → LLM can cite them

Output format:
The chain returns a structured dict (not just a string):
  {answer, sources: [{title, page_number, chunk_id, score}]}
This is what feeds the SourceCitations UI component.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.ingestion.embedder import get_embedder
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

RetrievalStrategy = Literal["dense", "hybrid", "hybrid_rerank"]

# System prompt — engineered to minimise hallucination and ensure groundedness
_SYSTEM_PROMPT = """You are a precise research paper assistant. Your task is to answer questions based ONLY on the provided research paper excerpts.

Rules you MUST follow:
1. Answer ONLY from the information in the provided context. Do not use external knowledge.
2. If the context does not contain enough information to answer the question, respond with: "I don't have enough information in the provided papers to answer this question."
3. When answering, reference the paper title and page number where you found the information.
4. Be concise but thorough. Aim for 2-4 paragraphs.
5. If multiple papers are relevant, synthesize information from them, citing each.
6. Do NOT make up citations, page numbers, or paper titles.

Format your answer as plain text with inline citations like: (Source: Paper Title, p.X)"""

_CONTEXT_TEMPLATE = """Context from research papers:

{context}

---
Question: {question}"""


def _build_llm(settings: object | None = None) -> ChatOpenAI:
    """
    Build the LLM client based on LLM_PROVIDER setting.

    Both NVIDIA NIM and Ollama expose an OpenAI-compatible /chat/completions
    endpoint, so LangChain's ChatOpenAI works for both without modification.
    """
    s = settings or get_settings()

    if s.llm_provider == "nvidia_nim":
        return ChatOpenAI(
            model=s.chat_model,
            openai_api_key=s.nvidia_api_key,
            openai_api_base=s.nvidia_base_url,
            temperature=0.1,      # low temperature for factual/grounded answers
            max_tokens=1024,
        )
    elif s.llm_provider == "ollama":
        return ChatOpenAI(
            model=s.ollama_model,
            openai_api_key="ollama",  # Ollama doesn't require a real key
            openai_api_base=f"{s.ollama_base_url}/v1",
            temperature=0.1,
            max_tokens=1024,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {s.llm_provider!r}")


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Paper: \"{chunk['paper_title']}\" (Page {chunk['page_number']})\n"
            f"{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def _build_sources(chunks: list[dict]) -> list[dict[str, Any]]:
    """Build the sources list returned alongside the answer."""
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "paper_title": chunk.get("paper_title", ""),
            "page_number": chunk.get("page_number"),
            "score": round(float(chunk.get("score", 0.0)), 4),
            "chunk_text": chunk.get("chunk_text", "")[:300],  # preview only
        }
        for chunk in chunks
    ]


async def rag_query(
    db: object,
    question: str,
    retrieval_strategy: RetrievalStrategy = "hybrid",
    embedding_model: str | None = None,
    chat_history: list[dict] | None = None,
    settings: object | None = None,
) -> dict[str, Any]:
    """
    End-to-end RAG query: retrieve → format → generate → return with sources.

    Args:
        db: async SQLAlchemy session
        question: the user's question
        retrieval_strategy: 'dense' | 'hybrid' | 'hybrid_rerank'
        embedding_model: override for the embedding model to use for retrieval
        chat_history: list of {role, content} dicts for conversational context
        settings: override for app settings (default: from env)

    Returns:
        {
            answer: str,
            sources: [{chunk_id, paper_title, page_number, score, chunk_text}],
            retrieval_strategy: str,
            embedding_model: str,
        }
    """
    s = settings or get_settings()
    emb_model = embedding_model or s.embedding_model

    # ── Step 1: Embed the query ─────────────────────────────────────────────
    embedder = get_embedder(emb_model, s)
    query_vec = await embedder.aembed([question])
    q_vec = query_vec[0]

    # ── Step 2: Retrieve relevant chunks ────────────────────────────────────
    if retrieval_strategy == "dense":
        chunks = await dense_search(db, q_vec, emb_model, top_k=5)

    elif retrieval_strategy == "hybrid":
        results = await hybrid_search(db, question, q_vec, emb_model, top_k=s.default_top_k)
        chunks = results[:5]

    elif retrieval_strategy == "hybrid_rerank":
        candidates = await hybrid_search(db, question, q_vec, emb_model, top_k=s.default_top_k)
        # Run cross-encoder reranking in a thread (CPU-bound)
        import asyncio  # noqa: PLC0415

        chunks = await asyncio.to_thread(rerank, question, candidates, s.reranker_top_n)

    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy!r}")

    logger.info(
        "RAG: strategy=%s model=%s retrieved=%d chunks",
        retrieval_strategy,
        emb_model,
        len(chunks),
    )

    # ── Step 3: Build the prompt ─────────────────────────────────────────────
    context_str = _format_context(chunks)
    user_content = _CONTEXT_TEMPLATE.format(context=context_str, question=question)

    messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Inject chat history for conversational memory
    if chat_history:
        for turn in chat_history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                from langchain_core.messages import AIMessage  # noqa: PLC0415

                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=user_content))

    # ── Step 4: Call the LLM ─────────────────────────────────────────────────
    llm = _build_llm(s)
    response = await llm.ainvoke(messages)
    answer = response.content

    # ── Step 5: Return structured result ─────────────────────────────────────
    return {
        "answer": answer,
        "sources": _build_sources(chunks),
        "retrieval_strategy": retrieval_strategy,
        "embedding_model": emb_model,
    }
