"""Hierarchical summarization with DistilBART."""

from __future__ import annotations

from src.evaluation import evaluate_summary
from src.model_manager import ModelBundle, hierarchical_summarize_for_model, load_pretrained_model

DISTILBART_MODEL = "sshleifer/distilbart-cnn-12-6"

LENGTH_PRESETS = {
    "short": {"chunk_max_new_tokens": 80, "final_max_new_tokens": 120},
    "medium": {"chunk_max_new_tokens": 120, "final_max_new_tokens": 200},
    "long": {"chunk_max_new_tokens": 160, "final_max_new_tokens": 280},
}


def load_summarizer() -> ModelBundle:
    """Load the DistilBART summarization model."""
    return load_pretrained_model(DISTILBART_MODEL)


def summarize_chunks(
    chunks: list[str],
    bundle: ModelBundle,
    *,
    length: str = "medium",
    max_chunks: int = 12,
    strategy: str = "hierarchical_distilbart",
) -> tuple[str, dict]:
    """
    Run hierarchical summarization over text chunks.

    Returns:
        (summary text, metrics dict)
    """
    preset = LENGTH_PRESETS.get(length, LENGTH_PRESETS["medium"])
    chunk_config = {
        "max_new_tokens": preset["chunk_max_new_tokens"],
        "num_beams": 4,
        "early_stopping": True,
    }
    final_config = {
        "max_new_tokens": preset["final_max_new_tokens"],
        "num_beams": 4,
        "early_stopping": True,
    }

    summary, stats = hierarchical_summarize_for_model(
        chunks,
        tokenizer=bundle.tokenizer,
        model=bundle.model,
        device=bundle.device,
        max_chunks=max_chunks,
        generation_config_chunk=chunk_config,
        generation_config_final=final_config,
    )

    source_text = "\n\n".join(chunks[:max_chunks])
    metrics = evaluate_summary(
        source_text,
        summary,
        float(stats.get("runtime_seconds", 0.0)),
        strategy=strategy,
    )
    metrics_dict = {
        "strategy": metrics.strategy,
        "source_words": metrics.source_words,
        "summary_words": metrics.summary_words,
        "compression_ratio": metrics.compression_ratio,
        "compression_percent": metrics.compression_percent,
        "runtime_seconds": metrics.runtime_seconds,
        "chunks_used": stats.get("chunks_used", 0),
        "generated_tokens": stats.get("generated_tokens", 0),
    }
    return summary, metrics_dict
