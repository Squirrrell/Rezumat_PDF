"""Convert domain objects to JSON-serializable dicts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.evaluation import DocumentMetrics, SummaryMetrics
from src.section_detector import PaperSections


def document_metrics_to_dict(metrics: DocumentMetrics) -> dict[str, Any]:
    return asdict(metrics)


def summary_metrics_to_dict(metrics: SummaryMetrics) -> dict[str, Any]:
    return asdict(metrics)


def sections_to_dict(sections: PaperSections, include_text: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "detected": sections.detected,
        "coverage_ratio": sections.coverage_ratio,
        "fallback_body": sections.fallback_body,
        "sections": {},
    }
    for key, text in sections.sections.items():
        if include_text:
            data["sections"][key] = text
        else:
            data["sections"][key] = {
                "characters": len(text),
                "words": len(text.split()),
            }
    return data
