"""Unit tests for comprehensive section-aware summarization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.comprehensive_summary import (
    _filter_duplicate_sentences,
    _should_skip_section,
    generate_comprehensive_summary,
)
from src.vector_store import VectorIndex


def test_should_skip_empty_and_not_specified():
    assert _should_skip_section("") is True
    assert _should_skip_section("Not clearly specified in the document.") is True
    assert _should_skip_section("The authors propose a transformer model.") is False


def test_filter_duplicate_sentences():
    seen: set[str] = set()
    first = _filter_duplicate_sentences(
        "The model uses attention. Results improved on benchmarks.",
        seen,
    )
    second = _filter_duplicate_sentences(
        "The model uses attention. A new dataset was introduced.",
        seen,
    )
    assert "attention" in first.lower()
    assert "attention" not in second.lower()
    assert "dataset" in second.lower()


@patch("src.comprehensive_summary.search")
def test_generate_comprehensive_summary_assembles_sections(mock_search):
    mock_search.return_value = [("Paper context about methods and results.", 0.1)]
    vector_index = VectorIndex(index=MagicMock(), chunks=["chunk"])
    embed_model = MagicMock()
    generator = MagicMock()

    def fake_generate(prompt, max_new_tokens=320):
        if "Key takeaways" in prompt:
            return "- Finding one\n- Finding two"
        if "Main idea" in prompt:
            return "The main idea is efficient training."
        return "Not clearly specified in the document."

    generator._generate = fake_generate

    text, metrics = generate_comprehensive_summary(
        vector_index,
        embed_model,
        generator,
        top_k=3,
        source_label="pdf",
    )

    assert "## Main idea" in text
    assert "efficient training" in text
    assert "## Key takeaways" in text
    assert metrics["sections_generated"] >= 2
    assert metrics["summary_words"] > 0
    assert metrics["source"] == "pdf"
