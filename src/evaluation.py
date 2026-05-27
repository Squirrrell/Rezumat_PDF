"""Intrinsic evaluation metrics for thesis experiments."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:  # pragma: no cover - optional dependency for the app
    rouge_scorer = None


@dataclass
class TimedResult:
    """Result of a timed operation."""

    label: str
    elapsed_seconds: float
    payload: str | None = None


@dataclass
class SummaryMetrics:
    """Metrics for a single summarization run."""

    strategy: str
    source_words: int
    summary_words: int
    compression_ratio: float
    compression_percent: float
    runtime_seconds: float


@dataclass
class ComparisonResult:
    """Summary text plus evaluation metrics for comparison mode."""

    strategy: str
    summary: str
    metrics: SummaryMetrics


def count_words(text: str) -> int:
    """Count words by whitespace splitting."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def compression_ratio_words(source_words: int, summary_words: int) -> float:
    """Return summary_words / source_words (0 if source is empty)."""
    if source_words <= 0:
        return 0.0
    return summary_words / source_words


def compression_ratio(original_text: str, summary_text: str) -> float:
    """Return summary_words / original_words (0 if original is empty)."""
    return compression_ratio_words(count_words(original_text), count_words(summary_text))


def compression_percent(source_words: int, summary_words: int) -> float:
    """Return percentage reduction: (1 - ratio) * 100."""
    ratio = compression_ratio_words(source_words, summary_words)
    return (1.0 - ratio) * 100.0


def evaluate_summary(
    source_text: str,
    summary: str,
    runtime_seconds: float,
    strategy: str = "full",
) -> SummaryMetrics:
    """Compute intrinsic metrics for one summarization."""
    source_words = count_words(source_text)
    summary_words = count_words(summary)
    ratio = compression_ratio_words(source_words, summary_words)
    percent = compression_percent(source_words, summary_words)
    return SummaryMetrics(
        strategy=strategy,
        source_words=source_words,
        summary_words=summary_words,
        compression_ratio=ratio,
        compression_percent=percent,
        runtime_seconds=runtime_seconds,
    )


def word_count(text: str) -> int:
    """Alias for count_words (kept for UI readability)."""
    return count_words(text)


def calculate_rouge(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute ROUGE-1/2/L F1 scores for a single prediction/reference pair.

    Requires `rouge-score` to be installed. If not installed, raises RuntimeError.
    """
    if rouge_scorer is None:
        raise RuntimeError(
            "ROUGE is unavailable. Install rouge-score (pip install rouge-score)."
        )
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference or "", prediction or "")
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def approx_token_count(text: str, tokenizer) -> int:
    """Approximate token count using the provided tokenizer."""
    if not text or not text.strip():
        return 0
    try:
        return int(len(tokenizer.encode(text, add_special_tokens=False)))
    except Exception:
        return 0


def compare_summaries(
    results: list[tuple[str, str, float, str]],
    source_texts: dict[str, str] | None = None,
    default_source: str = "",
) -> list[ComparisonResult]:
    """
    Evaluate multiple summarization strategies.

    Args:
        results: List of (strategy, summary, runtime_seconds, source_text).
                 If source_text is empty, uses default_source or source_texts[strategy].
        source_texts: Optional per-strategy source text for metrics.
        default_source: Fallback source when per-item source is empty.
    """
    comparison: list[ComparisonResult] = []
    source_texts = source_texts or {}

    for item in results:
        if len(item) == 4:
            strategy, summary, runtime, source = item
        else:
            strategy, summary, runtime = item
            source = ""

        if not source:
            source = source_texts.get(strategy, default_source)

        metrics = evaluate_summary(source, summary, runtime, strategy=strategy)
        comparison.append(
            ComparisonResult(strategy=strategy, summary=summary, metrics=metrics)
        )

    return comparison


def metrics_to_dict(metrics: SummaryMetrics) -> dict:
    """Convert metrics to a plain dict for display or storage."""
    return asdict(metrics)


def evaluate_qa(
    answer: str,
    sources: list[dict],
    runtime_seconds: float,
) -> dict:
    """Lightweight metrics for a Q&A interaction."""
    source_word_counts = [count_words(s.get("text", "")) for s in sources]
    avg_source_words = (
        sum(source_word_counts) / len(source_word_counts) if source_word_counts else 0
    )
    return {
        "answer_words": count_words(answer),
        "num_sources": len(sources),
        "avg_source_words": round(avg_source_words, 1),
        "runtime_seconds": runtime_seconds,
    }


def format_summary_export(
    summary: str,
    metrics: SummaryMetrics,
    paper_name: str = "",
    extra_sections: dict[str, str] | None = None,
) -> str:
    """Build plain-text export content for download."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "Research Paper Summary Export",
        "=============================",
        f"Paper: {paper_name or 'N/A'}",
        f"Strategy: {metrics.strategy}",
        f"Generated: {timestamp}",
        "",
        "--- Metrics ---",
        f"Source words: {metrics.source_words}",
        f"Summary words: {metrics.summary_words}",
        (
            f"Compression ratio: {metrics.compression_ratio:.4f} "
            f"({metrics.compression_percent:.1f}% reduction)"
        ),
        f"Runtime: {metrics.runtime_seconds:.2f} s",
        "",
        "--- Summary ---",
        summary,
    ]

    if extra_sections:
        lines.extend(["", "--- Detected sections (preview) ---"])
        for name, text in extra_sections.items():
            preview = text[:500] + ("..." if len(text) > 500 else "")
            lines.append(f"\n[{name}]\n{preview}")

    return "\n".join(lines)
