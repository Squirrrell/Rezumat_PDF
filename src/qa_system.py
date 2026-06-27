"""Question answering via retrieval and instruct models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from src.instruct_generator import QA_MAX_CONTEXT_CHARS
from src.llm_config import qa_max_new_tokens
from src.vector_store import VectorIndex, search

if TYPE_CHECKING:
    from src.instruct_generator import InstructGenerator

QA_REFUSAL_PHRASES = (
    "does not provide enough detail",
    "not clearly specified in the document",
    "available pdf text does not provide",
    "no relevant passages found",
)

MAX_QA_TOP_K = 10


def _dedupe_hits(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
    seen: set[str] = set()
    unique: list[tuple[str, float]] = []
    for text, score in hits:
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append((text, score))
    return unique


def _build_context(hits: list[tuple[str, float]]) -> str:
    parts: list[str] = []
    total = 0
    for text, _ in hits:
        chunk = text.strip()
        if not chunk:
            continue
        if total + len(chunk) + 2 > QA_MAX_CONTEXT_CHARS:
            remaining = QA_MAX_CONTEXT_CHARS - total
            if remaining > 200:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk) + 2
    return "\n\n".join(parts)


def _looks_like_refusal(answer: str) -> bool:
    normalized = answer.strip().lower()
    if not normalized:
        return True
    return any(phrase in normalized for phrase in QA_REFUSAL_PHRASES)


def answer_question(
    question: str,
    vector_index: VectorIndex,
    embed_model: SentenceTransformer,
    top_k: int = 5,
    instruct_generator: InstructGenerator | None = None,
    *,
    answer_length: str = "medium",
) -> tuple[str, list[dict]]:
    """
    Answer a question by retrieving relevant chunks and generating a response.

    Returns:
        Tuple of (answer text, list of source dicts with 'text' and 'score').
    """
    question = question.strip()
    if not question:
        return "", []

    if instruct_generator is None:
        raise RuntimeError("Instruct model is required for Q&A.")

    max_tokens = qa_max_new_tokens(answer_length)
    effective_top_k = min(max(1, top_k), len(vector_index.chunks))
    hits = _dedupe_hits(search(vector_index, question, embed_model, top_k=effective_top_k))
    if not hits:
        return "No relevant passages found in the document.", []

    context = _build_context(hits)
    answer = instruct_generator.generate_qa_answer(
        question,
        context,
        max_new_tokens=max_tokens,
    )

    if _looks_like_refusal(answer):
        expanded_k = min(MAX_QA_TOP_K, len(vector_index.chunks))
        if expanded_k > effective_top_k:
            expanded_hits = _dedupe_hits(
                search(vector_index, question, embed_model, top_k=expanded_k)
            )
            expanded_context = _build_context(expanded_hits)
            retry_answer = instruct_generator.generate_qa_answer(
                question,
                expanded_context,
                max_new_tokens=max_tokens,
            )
            if retry_answer.strip() and not _looks_like_refusal(retry_answer):
                answer = retry_answer
                hits = expanded_hits

    sources = [{"text": text, "score": score} for text, score in hits]
    return answer, sources
