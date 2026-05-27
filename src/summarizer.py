"""Hierarchical summarization with distilbart-cnn-12-6."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from src.instruct_generator import InstructGenerator
    from src.structured_summary import StructuredSummaryResult
    from src.vector_store import VectorIndex

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MAX_INPUT_TOKENS = 1024
COMBINE_THRESHOLD = 6000

# Re-export for backward compatibility
STRUCTURED_SECTIONS = [
    "Title / Topic",
    "Main idea",
    "Problem addressed",
    "Proposed method",
    "Dataset / experiments",
    "Results",
    "Limitations",
    "Conclusion",
    "Key takeaways",
]

SUMMARY_PRESETS: dict[str, dict[str, int]] = {
    "short": {"max_length": 220, "min_length": 80},
    "detailed": {"max_length": 512, "min_length": 150},
    "bullet points": {"max_length": 550, "min_length": 120},
}

INTERMEDIATE_PRESET = {"max_length": 200, "min_length": 50}


@dataclass
class GenerationSettings:
    """Parameters passed to model.generate for summarization."""

    max_length: int
    min_length: int
    max_new_tokens: int | None = None
    num_beams: int = 4
    no_repeat_ngram_size: int = 4
    repetition_penalty: float = 1.3
    length_penalty: float = 1.2
    early_stopping: bool = True


def get_device() -> str:
    """Return cuda if available, otherwise cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_summarizer() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, str]:
    """
    Load tokenizer and model.

    Raises:
        RuntimeError: If model loading fails.
    """
    device = get_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load summarization model '{MODEL_NAME}'. "
            f"Check your internet connection and Hugging Face cache. Details: {exc}"
        ) from exc


def get_summary_preset(
    summary_type: str,
    max_override: int | None = None,
    min_override: int | None = None,
) -> GenerationSettings:
    """Build GenerationSettings from summary type preset and optional sidebar overrides."""
    summary_type = summary_type.lower()
    preset = SUMMARY_PRESETS.get(summary_type, SUMMARY_PRESETS["short"])
    max_len = max_override if max_override is not None else preset["max_length"]
    min_len = min_override if min_override is not None else preset["min_length"]
    min_len = min(min_len, max_len - 1) if max_len > 1 else 1
    return GenerationSettings(max_length=max_len, min_length=min_len, max_new_tokens=max_len)


def get_intermediate_settings() -> GenerationSettings:
    """Settings for per-chunk and merge passes."""
    return GenerationSettings(
        max_length=INTERMEDIATE_PRESET["max_length"],
        min_length=INTERMEDIATE_PRESET["min_length"],
        max_new_tokens=INTERMEDIATE_PRESET["max_length"],
    )


def summarize_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    settings: GenerationSettings | None = None,
    max_length: int | None = None,
    min_length: int | None = None,
) -> str:
    """Summarize a single piece of text."""
    text = text.strip()
    if not text:
        return ""

    if settings is None:
        settings = GenerationSettings(
            max_length=max_length or INTERMEDIATE_PRESET["max_length"],
            min_length=min_length or INTERMEDIATE_PRESET["min_length"],
            max_new_tokens=max_length or INTERMEDIATE_PRESET["max_length"],
        )

    inputs = tokenizer(
        text,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    max_new = settings.max_new_tokens or settings.max_length
    safe_min = min(settings.min_length, max_new - 1)
    safe_min = max(safe_min, 1) if max_new > 1 else 1

    gen_kwargs: dict = {
        "max_new_tokens": max_new,
        "min_length": safe_min,
        "num_beams": settings.num_beams,
        "no_repeat_ngram_size": settings.no_repeat_ngram_size,
        "repetition_penalty": settings.repetition_penalty,
        "length_penalty": settings.length_penalty,
        "early_stopping": settings.early_stopping,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def format_as_bullets(text: str, max_bullets: int = 12) -> str:
    """Convert summary text into bullet points."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    bullets = [s.strip() for s in sentences if s.strip()]
    if not bullets:
        return text
    bullets = bullets[:max_bullets]
    return "\n".join(f"- {b}" for b in bullets)


def _legacy_structured_distilbart(
    combined_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
) -> str:
    """Legacy structured summary using same context (distilbart). Prefer section-aware."""
    from src.structured_summary import MISSING_TEXT, STRUCTURED_SECTIONS as SECTIONS

    section_settings = GenerationSettings(
        max_length=110,
        min_length=30,
        max_new_tokens=110,
    )
    parts: list[str] = []
    for section in SECTIONS:
        prompt = (
            f"Summarize only the following aspect of this research paper: {section}. "
            f"Use only facts from the text.\n\n{combined_text}"
        )
        section_text = summarize_text(
            prompt, tokenizer, model, device, settings=section_settings
        )
        if not section_text:
            section_text = MISSING_TEXT
        parts.append(f"{section}:\n{section_text}")
    return "\n\n".join(parts)


def _merge_combined_summaries(
    combined: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    intermediate_settings: GenerationSettings,
    skip_aggressive_compress: bool,
) -> str:
    """Optionally compress very long combined chunk summaries."""
    if len(combined) <= COMBINE_THRESHOLD:
        return combined

    if skip_aggressive_compress:
        mid = len(combined) // 2
        parts = [combined[:mid], combined[mid:]]
        merged_parts = [
            summarize_text(part, tokenizer, model, device, settings=intermediate_settings)
            for part in parts
            if part.strip()
        ]
        return "\n\n".join(p for p in merged_parts if p)

    return summarize_text(
        combined,
        tokenizer,
        model,
        device,
        settings=intermediate_settings,
    )


def hierarchical_summarize(
    chunks: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    summary_type: str = "short",
    max_chunks: int = 12,
    settings: GenerationSettings | None = None,
    detailed_structured: bool = False,
    section_aware_structured: bool = False,
    intermediate_settings: GenerationSettings | None = None,
    vector_index: VectorIndex | None = None,
    embed_model: SentenceTransformer | None = None,
    instruct_generator: InstructGenerator | None = None,
    structured_top_k: int = 5,
) -> str | tuple[str, StructuredSummaryResult]:
    """
    Hierarchical summarization with optional section-aware structured output.

    When section_aware_structured is True and vector_index is provided, returns
    (body, StructuredSummaryResult). Otherwise returns summary str only.
    """
    if not chunks:
        return ""

    summary_type = summary_type.lower()
    if summary_type not in SUMMARY_PRESETS:
        summary_type = "short"

    if settings is None:
        settings = get_summary_preset(summary_type)
    if intermediate_settings is None:
        intermediate_settings = get_intermediate_settings()

    if detailed_structured and section_aware_structured:
        if vector_index is None or embed_model is None or instruct_generator is None:
            raise ValueError(
                "Section-aware structured summary requires vector_index, "
                "embed_model, and instruct_generator."
            )
        from src.structured_summary import generate_structured_summary

        result = generate_structured_summary(
            vector_index,
            embed_model,
            instruct_generator,
            top_k=structured_top_k,
        )
        return result.body, result

    skip_compress = summary_type == "detailed" or detailed_structured

    selected = chunks[:max_chunks]
    chunk_summaries: list[str] = []

    for chunk in selected:
        summary = summarize_text(
            chunk,
            tokenizer,
            model,
            device,
            settings=intermediate_settings,
        )
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return ""

    combined = "\n\n".join(chunk_summaries)
    combined = _merge_combined_summaries(
        combined,
        tokenizer,
        model,
        device,
        intermediate_settings,
        skip_aggressive_compress=skip_compress,
    )

    if detailed_structured:
        return _legacy_structured_distilbart(combined, tokenizer, model, device)

    final = summarize_text(
        combined,
        tokenizer,
        model,
        device,
        settings=settings,
    )

    if summary_type == "bullet points":
        return format_as_bullets(final, max_bullets=12)

    return final
