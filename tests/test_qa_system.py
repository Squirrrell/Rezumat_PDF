"""Unit tests for Q&A retrieval and answer generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.qa_system import (
    _build_context,
    _dedupe_hits,
    _looks_like_refusal,
    answer_question,
)
from src.vector_store import VectorIndex


def test_dedupe_hits_removes_duplicates():
    hits = [
        ("same chunk", 0.1),
        ("same chunk", 0.2),
        ("other chunk", 0.3),
    ]
    result = _dedupe_hits(hits)
    assert len(result) == 2
    assert result[0][0] == "same chunk"
    assert result[1][0] == "other chunk"


def test_build_context_respects_max_chars():
    hits = [("x" * 5000, 0.1), ("y" * 5000, 0.2)]
    context = _build_context(hits)
    assert len(context) <= 8000


def test_looks_like_refusal_detects_canned_phrases():
    assert _looks_like_refusal("The available PDF text does not provide enough detail.") is True
    assert _looks_like_refusal("The method uses transformers and attention.") is False
    assert _looks_like_refusal("") is True


@patch("src.qa_system.search")
def test_answer_question_retries_with_more_chunks_on_refusal(mock_search):
    mock_search.side_effect = [
        [("partial context about methods.", 0.2)],
        [
            ("partial context about methods.", 0.2),
            ("results show improved accuracy on benchmarks.", 0.3),
        ],
    ]

    generator = MagicMock()
    generator.generate_qa_answer.side_effect = [
        "The available PDF text does not provide enough detail to answer fully.",
        "The paper reports improved accuracy on standard benchmarks.",
    ]

    index = VectorIndex(index=MagicMock(), chunks=["a", "b", "c", "d", "e", "f"])
    embed_model = MagicMock()

    answer, sources = answer_question(
        "What were the results?",
        index,
        embed_model,
        top_k=3,
        instruct_generator=generator,
        answer_length="medium",
    )

    assert "improved accuracy" in answer.lower()
    assert mock_search.call_count == 2
    assert mock_search.call_args_list[1].kwargs["top_k"] == 6
    assert len(sources) == 2
    assert generator.generate_qa_answer.call_count == 2
    assert generator.generate_qa_answer.call_args_list[0].kwargs["max_new_tokens"] == 400


@patch("src.qa_system.search")
def test_answer_question_returns_helpful_answer_without_retry(mock_search):
    mock_search.return_value = [("The model uses self-attention over tokens.", 0.1)]

    generator = MagicMock()
    generator.generate_qa_answer.return_value = (
        "The model applies self-attention over input tokens."
    )

    embeddings = np.zeros((1, 384), dtype="float32")
    faiss_index = MagicMock()
    faiss_index.search.return_value = (np.array([[0.0]]), np.array([[0]]))
    index = VectorIndex(index=faiss_index, chunks=["The model uses self-attention over tokens."])
    embed_model = MagicMock()
    embed_model.encode.return_value = embeddings

    answer, sources = answer_question(
        "What mechanism is used?",
        index,
        embed_model,
        top_k=5,
        instruct_generator=generator,
    )

    assert "self-attention" in answer.lower()
    assert len(sources) == 1
    assert generator.generate_qa_answer.call_count == 1
