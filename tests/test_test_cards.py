"""Unit tests for test card generation and LLM judge verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.test_cards import (
    TestCard,
    _build_fallback_context,
    _parse_json_block,
    generate_test_cards,
    judge_answer_with_llm,
    verify_test_cards,
)
from src.vector_store import VectorIndex


def test_parse_json_block_strips_code_fences():
    raw = '```json\n{"question": "Q?", "reference_answer": "A.", "key_phrases": ["a", "b", "c"]}\n```'
    parsed = _parse_json_block(raw)
    assert parsed is not None
    assert parsed["question"] == "Q?"


def test_build_fallback_context_rotates_chunks():
    index = VectorIndex(index=MagicMock(), chunks=["chunk-a", "chunk-b", "chunk-c"])
    ctx0 = _build_fallback_context(index, 0)
    ctx1 = _build_fallback_context(index, 1)
    assert "chunk-a" in ctx0
    assert "chunk-b" in ctx0
    assert ctx0 != ctx1


@patch("src.test_cards.search", return_value=[])
def test_generate_retries_until_target_count(mock_search):
    vector_index = VectorIndex(index=MagicMock(), chunks=["paper text about methods and results"])
    embed_model = MagicMock()
    generator = MagicMock()

    call_count = {"n": 0}

    def fake_generate(prompt, max_new_tokens=400):
        call_count["n"] += 1
        n = call_count["n"]
        if n == 1:
            return "not json"
        return (
            f'{{"question": "Question {n}?", "reference_answer": "Answer {n}.", '
            f'"key_phrases": ["p1", "p2", "p3"]}}'
        )

    generator._generate = fake_generate

    cards = generate_test_cards(vector_index, embed_model, generator, num_cards=3)

    assert len(cards) == 3
    assert len({c.question for c in cards}) == 3
    assert mock_search.called


def test_judge_answer_with_llm_valid_json():
    card = TestCard(
        id="c1",
        question="What method was used?",
        reference_answer="They used transformers.",
        key_phrases=["transformers", "attention", "benchmark"],
    )
    generator = MagicMock()
    generator._generate.return_value = (
        '{"score": 85, "matched_phrases": ["transformers", "attention"], '
        '"missed_phrases": ["benchmark"], "feedback": "Good coverage of the method."}'
    )

    result = judge_answer_with_llm(card, "They proposed a transformer with attention.", generator)

    assert result is not None
    assert result["score"] == 85
    assert result["matched_phrases"] == ["transformers", "attention"]
    assert result["missed_phrases"] == ["benchmark"]
    assert "Good coverage" in result["feedback"]


def test_judge_answer_with_llm_invalid_json_returns_none():
    card = TestCard(
        id="c1",
        question="Q?",
        reference_answer="A.",
        key_phrases=["a", "b", "c"],
    )
    generator = MagicMock()
    generator._generate.return_value = "not valid json"

    assert judge_answer_with_llm(card, "some answer here", generator) is None


def test_verify_test_cards_llm_success():
    card = TestCard(
        id="c1",
        question="What is the main idea?",
        reference_answer="Neural networks for NLP.",
        key_phrases=["neural networks", "NLP"],
    )
    generator = MagicMock()
    generator._generate.return_value = (
        '{"score": 90, "matched_phrases": ["neural networks", "NLP"], '
        '"missed_phrases": [], "feedback": "Excellent answer."}'
    )

    results, total = verify_test_cards([card], {"c1": "Neural networks applied to NLP tasks."}, generator=generator)

    assert len(results) == 1
    assert results[0].score == 90
    assert results[0].scoring_method == "llm"
    assert results[0].judge_feedback == "Excellent answer."
    assert results[0].reference_answer == "Neural networks for NLP."
    assert total == 90


def test_verify_test_cards_llm_failure_falls_back_to_keyword():
    card = TestCard(
        id="c1",
        question="What is the main idea?",
        reference_answer="Neural networks for NLP.",
        key_phrases=["neural networks", "NLP"],
    )
    generator = MagicMock()
    generator._generate.return_value = "garbage output"

    answer = "The paper uses neural networks for NLP problems."
    results, total = verify_test_cards([card], {"c1": answer}, generator=generator)

    assert len(results) == 1
    assert results[0].scoring_method == "keyword_fallback"
    assert results[0].score == 100
    assert total == 100


def test_verify_test_cards_empty_answer_skips_llm():
    card = TestCard(
        id="c1",
        question="What is the main idea?",
        reference_answer="Neural networks for NLP.",
        key_phrases=["neural networks", "NLP"],
    )
    generator = MagicMock()

    results, total = verify_test_cards([card], {"c1": ""}, generator=generator)

    assert len(results) == 1
    assert results[0].score == 0
    assert results[0].scoring_method == "llm"
    generator._generate.assert_not_called()
    assert total == 0
