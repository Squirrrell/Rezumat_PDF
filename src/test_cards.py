"""Comprehension test cards: generate questions from PDF context and score answers."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from src.summary_fields import FIELD_RETRIEVAL_QUERIES
from src.vector_store import VectorIndex, search

if TYPE_CHECKING:
    from src.instruct_generator import InstructGenerator

FALLBACK_QUERIES = [
    "introduction methods results",
    "abstract contribution evaluation",
    "main idea contribution method results",
]

MIN_ANSWER_CHARS = 10
RETRIEVAL_TOP_K = 5
MAX_ATTEMPTS_MULTIPLIER = 4

CARD_GENERATION_PROMPT = """You are creating a comprehension quiz about a scientific paper.
Use ONLY the context below. Do not invent facts not supported by the context.

Context:
{context}

Task:
Create ONE clear comprehension question about this context, a short reference answer (2-4 sentences),
and 4-6 key phrases that a good student answer should mention (short phrases, 1-4 words, paper-specific).

Respond with ONLY valid JSON in this exact shape:
{{"question": "...", "reference_answer": "...", "key_phrases": ["phrase1", "phrase2", "phrase3", "phrase4"]}}
"""

JUDGE_PROMPT = """You are grading a student's comprehension of a scientific paper.

Question:
{question}

Reference answer:
{reference_answer}

Key ideas to cover:
{key_phrases_list}

Student answer:
{user_answer}

Rules:
- Score semantic correctness 0-100, not exact wording.
- Partial credit when some key ideas are present but incomplete.
- Score 0 if empty, off-topic, or contradicts the reference.
- matched_phrases / missed_phrases must use items from the key ideas list only.
- feedback: one sentence, max 20 words.

Respond with ONLY valid JSON:
{{"score": 0, "matched_phrases": [], "missed_phrases": [], "feedback": "..."}}
"""


@dataclass
class TestCard:
    """Server-side test card with hidden scoring data."""

    id: str
    question: str
    reference_answer: str
    key_phrases: list[str]


@dataclass
class CardScore:
    """Per-card verification result."""

    card_id: str
    score: int
    matched_phrases: list[str]
    missed_phrases: list[str]
    judge_feedback: str | None = None
    scoring_method: str = "llm"
    reference_answer: str | None = None


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped JSON in them."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _parse_json_block(text: str) -> dict | None:
    """Extract and parse a JSON object from model output."""
    text = _strip_code_fences(text.strip())
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_phrases(raw: list) -> list[str]:
    phrases: list[str] = []
    for item in raw:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                phrases.append(cleaned)
    return phrases[:6]


def _build_context(hits: list[tuple[str, float]]) -> str:
    parts = [text.strip() for text, _ in hits if text.strip()]
    return "\n\n".join(parts)[:6000]


def _build_query_pool() -> list[str]:
    pool = list(FIELD_RETRIEVAL_QUERIES.values())
    pool.extend(FALLBACK_QUERIES)
    return pool


def _build_fallback_context(vector_index: VectorIndex, attempt_index: int) -> str:
    """Use rotating chunk slices when semantic search returns no hits."""
    chunks = vector_index.chunks
    if not chunks:
        return ""

    chunk_count = min(3, len(chunks))
    start = attempt_index % len(chunks)
    selected = [chunks[(start + i) % len(chunks)] for i in range(chunk_count)]
    parts = [text.strip() for text in selected if text.strip()]
    return "\n\n".join(parts)[:6000]


def _card_from_parsed(parsed: dict, min_key_phrases: int) -> TestCard | None:
    question = str(parsed.get("question", "")).strip()
    reference_answer = str(parsed.get("reference_answer", "")).strip()
    key_phrases = _normalize_phrases(parsed.get("key_phrases", []))

    if not question or not reference_answer or len(key_phrases) < min_key_phrases:
        return None

    return TestCard(
        id=str(uuid.uuid4()),
        question=question,
        reference_answer=reference_answer,
        key_phrases=key_phrases,
    )


def _generate_one_card(
    generator: InstructGenerator,
    vector_index: VectorIndex,
    embed_model: SentenceTransformer,
    query: str,
    attempt_index: int = 0,
) -> TestCard | None:
    hits = search(vector_index, query, embed_model, top_k=RETRIEVAL_TOP_K)
    if hits:
        context = _build_context(hits)
    else:
        context = _build_fallback_context(vector_index, attempt_index)

    if not context:
        return None

    prompt = CARD_GENERATION_PROMPT.format(context=context)
    attempts = (
        (prompt, 3),
        (prompt + "\n\nReturn only the JSON object, no other text.", 2),
    )

    for attempt_prompt, min_key_phrases in attempts:
        raw = generator._generate(attempt_prompt, max_new_tokens=400)
        parsed = _parse_json_block(raw)
        if not parsed:
            continue
        card = _card_from_parsed(parsed, min_key_phrases)
        if card is not None:
            return card

    return None


def generate_test_cards(
    vector_index: VectorIndex,
    embed_model: SentenceTransformer,
    generator: InstructGenerator,
    num_cards: int = 5,
) -> list[TestCard]:
    """Generate comprehension cards using diverse retrieval queries with retries."""
    query_pool = _build_query_pool()
    max_attempts = num_cards * MAX_ATTEMPTS_MULTIPLIER
    cards: list[TestCard] = []
    seen_questions: set[str] = set()
    attempts = 0

    while len(cards) < num_cards and attempts < max_attempts:
        query = query_pool[attempts % len(query_pool)]
        card = _generate_one_card(
            generator,
            vector_index,
            embed_model,
            query,
            attempt_index=attempts,
        )
        attempts += 1

        if card is None:
            continue

        normalized_q = card.question.lower().strip()
        if normalized_q in seen_questions:
            continue

        seen_questions.add(normalized_q)
        cards.append(card)

    return cards


def score_answer(user_answer: str, key_phrases: list[str]) -> tuple[int, list[str], list[str]]:
    """Score a free-text answer by keyword/key-phrase overlap (0-100)."""
    answer = user_answer.strip()
    if len(answer) < MIN_ANSWER_CHARS or not key_phrases:
        return 0, [], list(key_phrases)

    normalized = answer.lower()
    matched = [p for p in key_phrases if p.lower() in normalized]
    missed = [p for p in key_phrases if p.lower() not in normalized]
    score = round(100 * len(matched) / len(key_phrases))
    return score, matched, missed


def _filter_phrases_to_key_list(raw: list, key_phrases: list[str]) -> list[str]:
    """Keep only phrases that appear in the card's key_phrases list."""
    allowed = {p.lower(): p for p in key_phrases}
    result: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        canonical = allowed.get(cleaned.lower())
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def judge_answer_with_llm(
    card: TestCard,
    user_answer: str,
    generator: InstructGenerator,
) -> dict | None:
    """Grade a student answer with the instruct model. Returns None on parse failure."""
    key_phrases_list = "\n".join(f"- {p}" for p in card.key_phrases)
    prompt = JUDGE_PROMPT.format(
        question=card.question,
        reference_answer=card.reference_answer,
        key_phrases_list=key_phrases_list,
        user_answer=user_answer.strip(),
    )
    raw = generator._generate(prompt, max_new_tokens=180)
    parsed = _parse_json_block(raw)
    if not parsed:
        return None

    try:
        score = int(parsed.get("score", 0))
    except (TypeError, ValueError):
        return None

    score = max(0, min(100, score))
    matched = _filter_phrases_to_key_list(parsed.get("matched_phrases", []), card.key_phrases)
    missed = _filter_phrases_to_key_list(parsed.get("missed_phrases", []), card.key_phrases)

    if not missed:
        missed = [p for p in card.key_phrases if p not in matched]

    feedback = str(parsed.get("feedback", "")).strip() or None

    return {
        "score": score,
        "matched_phrases": matched,
        "missed_phrases": missed,
        "feedback": feedback,
    }


def verify_test_cards(
    cards: list[TestCard],
    user_answers: dict[str, str],
    *,
    generator: InstructGenerator | None = None,
) -> tuple[list[CardScore], int]:
    """Verify user answers against stored cards; return per-card scores and total average."""
    results: list[CardScore] = []

    for card in cards:
        user_answer = user_answers.get(card.id, "")

        if len(user_answer.strip()) < MIN_ANSWER_CHARS:
            results.append(
                CardScore(
                    card_id=card.id,
                    score=0,
                    matched_phrases=[],
                    missed_phrases=list(card.key_phrases),
                    judge_feedback="Answer too short or empty.",
                    scoring_method="llm",
                    reference_answer=card.reference_answer,
                )
            )
            continue

        judged = None
        if generator is not None:
            judged = judge_answer_with_llm(card, user_answer, generator)

        if judged is not None:
            results.append(
                CardScore(
                    card_id=card.id,
                    score=judged["score"],
                    matched_phrases=judged["matched_phrases"],
                    missed_phrases=judged["missed_phrases"],
                    judge_feedback=judged["feedback"],
                    scoring_method="llm",
                    reference_answer=card.reference_answer,
                )
            )
            continue

        score, matched, missed = score_answer(user_answer, card.key_phrases)
        results.append(
            CardScore(
                card_id=card.id,
                score=score,
                matched_phrases=matched,
                missed_phrases=missed,
                judge_feedback=None,
                scoring_method="keyword_fallback",
                reference_answer=card.reference_answer,
            )
        )

    if not results:
        return [], 0

    total_score = round(sum(r.score for r in results) / len(results))
    return results, total_score


def card_to_public_dict(card: TestCard) -> dict:
    """Expose only client-safe fields."""
    return {"id": card.id, "question": card.question}


def card_to_session_dict(card: TestCard) -> dict:
    """Full card data stored server-side."""
    return {
        "id": card.id,
        "question": card.question,
        "reference_answer": card.reference_answer,
        "key_phrases": card.key_phrases,
    }


def card_from_session_dict(data: dict) -> TestCard:
    return TestCard(
        id=data["id"],
        question=data["question"],
        reference_answer=data["reference_answer"],
        key_phrases=list(data.get("key_phrases", [])),
    )


def score_to_dict(score: CardScore) -> dict:
    return {
        "card_id": score.card_id,
        "score": score.score,
        "matched_phrases": score.matched_phrases,
        "missed_phrases": score.missed_phrases,
        "judge_feedback": score.judge_feedback,
        "scoring_method": score.scoring_method,
        "reference_answer": score.reference_answer,
    }
