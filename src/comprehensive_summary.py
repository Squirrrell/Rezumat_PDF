"""Section-aware comprehensive summarization (~1 page) via instruct models."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from src.evaluation import count_words
from src.summary_fields import COMPREHENSIVE_SECTIONS, FIELD_RETRIEVAL_QUERIES
from src.vector_store import VectorIndex, search

if TYPE_CHECKING:
    from src.instruct_generator import InstructGenerator

NOT_SPECIFIED = "Not clearly specified in the document."

COMPREHENSIVE_FIELD_PROMPT = """You are analyzing a scientific paper.
Answer ONLY using the provided context.
Do not invent information.
If the context does not contain enough information, say:
"Not clearly specified in the document."

Context:
{context}

Task:
Write the "{field_name}" section of a comprehensive paper summary.

Rules:
- Be specific and paper-focused.
- Use 4-6 sentences with concrete details from the context.
- Do not repeat generic statements or content from other sections.
- Focus only on this field.

Answer:"""

COMPREHENSIVE_KEY_TAKEAWAYS_PROMPT = """You are analyzing a scientific paper.
Answer ONLY using the provided context.
Write 4-6 concrete key takeaways as bullet points (one per line, starting with "- ").
Do not invent information.
Avoid repeating the same point.

Context:
{context}

Key takeaways:"""

MAX_CONTEXT_CHARS = 6000

def _build_context(hits: list[tuple[str, float]]) -> str:
    parts = [text.strip() for text, _ in hits if text.strip()]
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def _normalize_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.strip().lower())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _filter_duplicate_sentences(text: str, seen: set[str]) -> str:
    """Drop sentences already used in prior sections."""
    if not text.strip():
        return ""

    lines = text.splitlines()
    if lines and all(line.strip().startswith("-") for line in lines if line.strip()):
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            norm = _normalize_sentence(stripped.lstrip("- ").strip())
            if norm in seen:
                continue
            seen.add(norm)
            kept.append(stripped)
        return "\n".join(kept)

    kept_sentences: list[str] = []
    for sentence in _split_sentences(text):
        norm = _normalize_sentence(sentence)
        if norm in seen:
            continue
        seen.add(norm)
        kept_sentences.append(sentence)
    return " ".join(kept_sentences)


def _should_skip_section(text: str) -> bool:
    cleaned = text.strip().lower()
    if not cleaned:
        return True
    if cleaned == NOT_SPECIFIED.lower():
        return True
    if cleaned.startswith("not clearly specified"):
        return True
    return False


def _generate_section_text(
    generator: InstructGenerator,
    field_name: str,
    context: str,
) -> str:
    context = context.strip()[:MAX_CONTEXT_CHARS]
    if not context:
        return ""

    if field_name == "Key takeaways":
        prompt = COMPREHENSIVE_KEY_TAKEAWAYS_PROMPT.format(context=context)
    else:
        prompt = COMPREHENSIVE_FIELD_PROMPT.format(context=context, field_name=field_name)

    return generator._generate(prompt, max_new_tokens=320).strip()


def generate_comprehensive_summary(
    vector_index: VectorIndex,
    embed_model: SentenceTransformer,
    generator: InstructGenerator,
    *,
    top_k: int = 5,
    source_label: str = "pdf",
) -> tuple[str, dict]:
    """
    Build a multi-section comprehensive summary (~1 page) using retrieval + instruct model.

    Returns:
        (markdown_text, metrics_dict)
    """
    t0 = time.perf_counter()
    sections_out: list[str] = []
    seen_sentences: set[str] = set()
    sections_generated = 0

    for field_name in COMPREHENSIVE_SECTIONS:
        query = FIELD_RETRIEVAL_QUERIES.get(field_name, field_name)
        hits = search(vector_index, query, embed_model, top_k=top_k)
        context = _build_context(hits)
        if not context:
            continue

        raw = _generate_section_text(generator, field_name, context)
        if _should_skip_section(raw):
            continue

        filtered = _filter_duplicate_sentences(raw, seen_sentences)
        if _should_skip_section(filtered):
            continue

        sections_out.append(f"## {field_name}\n\n{filtered}")
        sections_generated += 1

    full_text = "\n\n".join(sections_out).strip()
    runtime = time.perf_counter() - t0
    summary_words = count_words(full_text)

    metrics = {
        "strategy": f"comprehensive_structured_{source_label}",
        "source": source_label,
        "summary_words": summary_words,
        "sections_generated": sections_generated,
        "runtime_seconds": runtime,
    }
    return full_text, metrics
