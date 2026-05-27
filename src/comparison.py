"""Three-way summarization comparison for thesis experiments."""

import time
from typing import Callable

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.evaluation import ComparisonResult, compare_summaries
from src.section_detector import (
    PaperSections,
    get_section_text,
    intro_conclusion_fallback,
)
from src.summarizer import GenerationSettings, hierarchical_summarize
from src.text_utils import chunk_text
from src.vector_store import VectorIndex, search

COMPARISON_RETRIEVAL_QUERY = (
    "main contributions methodology experiments results conclusion"
)

STRATEGY_FULL = "full"
STRATEGY_INTRO_CONCLUSION = "intro_conclusion"
STRATEGY_RETRIEVED = "retrieved_only"


def _summarize_chunks(
    chunks: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    summary_type: str,
    max_chunks: int,
    settings: GenerationSettings | None = None,
    detailed_structured: bool = False,
) -> str:
    if not chunks:
        return ""
    return hierarchical_summarize(
        chunks,
        tokenizer,
        model,
        device,
        summary_type=summary_type,
        max_chunks=max_chunks,
        settings=settings,
        detailed_structured=detailed_structured,
    )


def run_comparison(
    cleaned_text: str,
    chunks: list[str],
    sections: PaperSections,
    vector_index: VectorIndex | None,
    embed_model: SentenceTransformer,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    summary_type: str,
    max_chunks: int,
    chunk_size: int,
    overlap: int,
    retrieval_top_k: int,
    settings: GenerationSettings | None = None,
    detailed_structured: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[ComparisonResult], dict[str, float]]:
    """
    Run all three comparison strategies sequentially.

    Returns:
        List of ComparisonResult and runtime dict keyed by strategy.
    """
    results: list[tuple[str, str, float, str]] = []
    runtimes: dict[str, float] = {}

    # --- Full document ---
    if on_progress:
        on_progress("Summarizing full document...")
    t0 = time.perf_counter()
    full_chunks = chunks[:max_chunks]
    full_summary = _summarize_chunks(
        full_chunks,
        tokenizer,
        model,
        device,
        summary_type,
        max_chunks,
        settings=settings,
        detailed_structured=detailed_structured,
    )
    runtimes[f"comparison_{STRATEGY_FULL}"] = time.perf_counter() - t0
    results.append(
        (STRATEGY_FULL, full_summary, runtimes[f"comparison_{STRATEGY_FULL}"], cleaned_text)
    )

    # --- Introduction + conclusion ---
    if on_progress:
        on_progress("Summarizing introduction + conclusion...")
    intro_text = get_section_text(sections, "introduction", "conclusion")
    if not intro_text.strip():
        intro_text = intro_conclusion_fallback(cleaned_text)

    intro_chunks = chunk_text(intro_text, chunk_size=chunk_size, overlap=overlap)
    t0 = time.perf_counter()
    intro_summary = _summarize_chunks(
        intro_chunks,
        tokenizer,
        model,
        device,
        summary_type,
        max_chunks,
        settings=settings,
        detailed_structured=detailed_structured,
    )
    runtimes[f"comparison_{STRATEGY_INTRO_CONCLUSION}"] = time.perf_counter() - t0
    results.append(
        (
            STRATEGY_INTRO_CONCLUSION,
            intro_summary,
            runtimes[f"comparison_{STRATEGY_INTRO_CONCLUSION}"],
            intro_text,
        )
    )

    # --- Retrieved chunks only ---
    if on_progress:
        on_progress("Summarizing retrieved chunks...")
    retrieved_text = ""
    retrieved_chunks: list[str] = []

    if vector_index is not None and vector_index.chunks:
        hits = search(
            vector_index,
            COMPARISON_RETRIEVAL_QUERY,
            embed_model,
            top_k=retrieval_top_k,
        )
        retrieved_chunks = [text for text, _ in hits]
        retrieved_text = "\n\n".join(retrieved_chunks)
    else:
        retrieved_chunks = chunks[:retrieval_top_k]
        retrieved_text = "\n\n".join(retrieved_chunks)

    t0 = time.perf_counter()
    retrieved_summary = _summarize_chunks(
        retrieved_chunks,
        tokenizer,
        model,
        device,
        summary_type,
        max_chunks=min(max_chunks, len(retrieved_chunks) or max_chunks),
        settings=settings,
        detailed_structured=detailed_structured,
    )
    runtimes[f"comparison_{STRATEGY_RETRIEVED}"] = time.perf_counter() - t0
    results.append(
        (
            STRATEGY_RETRIEVED,
            retrieved_summary,
            runtimes[f"comparison_{STRATEGY_RETRIEVED}"],
            retrieved_text,
        )
    )

    comparison_results = compare_summaries(results)
    return comparison_results, runtimes
