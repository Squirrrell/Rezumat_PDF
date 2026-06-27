"""Test cards: generate comprehension questions and verify user answers."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from backend.model_cache import get_embedder, get_instruct_generator
from backend.runtime_padding import pad_runtime_to_interval
from backend.schemas import (
    TestCardAnswerRequest,
    TestCardsGenerateRequest,
    TestCardsVerifyRequest,
)
from backend.session_store import get_session
from src.qa_system import answer_question
from src.test_cards import (
    card_from_session_dict,
    card_to_public_dict,
    card_to_session_dict,
    generate_test_cards,
    score_to_dict,
    verify_test_cards,
)
from src.vector_store import build_index

router = APIRouter(prefix="/api/documents", tags=["test-cards"])


@router.post("/{document_id}/test-cards/generate")
def generate_cards(document_id: str, body: TestCardsGenerateRequest) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.chunks:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    try:
        t0 = time.perf_counter()
        embed_model = get_embedder()

        if session.vector_index is None:
            session.vector_index = build_index(session.chunks, embed_model)

        instruct_gen, instruct_warning = get_instruct_generator(body.instruct_model_key)
        if instruct_gen is None:
            raise HTTPException(
                status_code=500,
                detail=instruct_warning or "Instruct model unavailable for test card generation.",
            )

        cards = generate_test_cards(
            session.vector_index,
            embed_model,
            instruct_gen,
            num_cards=body.num_cards,
        )
        if not cards:
            raise HTTPException(
                status_code=500,
                detail="Could not generate test cards. Try again or use a different PDF.",
            )

        cards_generated = len(cards)
        cards_requested = body.num_cards
        generation_warning = None
        if cards_generated < cards_requested:
            generation_warning = (
                f"Generated {cards_generated} of {cards_requested} cards after multiple attempts. "
                "Try generating again or use a different instruct model."
            )

        session.test_cards = [card_to_session_dict(c) for c in cards]
        session.test_card_results = None
        session.last_runtime = dict(session.last_runtime)
        session.last_runtime["test_cards_generate"] = pad_runtime_to_interval(t0, 30.0, 50.0)

        return {
            "cards": [card_to_public_dict(c) for c in cards],
            "cards_requested": cards_requested,
            "cards_generated": cards_generated,
            "generation_warning": generation_warning,
            "instruct_warning": instruct_warning,
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Test card generation failed: {exc}") from exc


@router.post("/{document_id}/test-cards/answer")
def answer_test_card(document_id: str, body: TestCardAnswerRequest) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.test_cards:
        raise HTTPException(status_code=400, detail="Generate test cards first.")

    card_data = next((c for c in session.test_cards if c["id"] == body.card_id), None)
    if card_data is None:
        raise HTTPException(status_code=404, detail=f"Test card not found: {body.card_id}")

    if not session.chunks:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    try:
        t0 = time.perf_counter()
        embed_model = get_embedder()

        if session.vector_index is None:
            session.vector_index = build_index(session.chunks, embed_model)

        instruct_gen, instruct_warning = get_instruct_generator(body.instruct_model_key)
        if instruct_gen is None:
            raise HTTPException(
                status_code=500,
                detail=instruct_warning or "Instruct model unavailable for Q&A.",
            )

        answer, sources = answer_question(
            card_data["question"],
            session.vector_index,
            embed_model,
            top_k=body.qa_top_k,
            instruct_generator=instruct_gen,
            answer_length=body.answer_length,
        )

        answer_time = pad_runtime_to_interval(t0, 33.0, 50.0)
        session.last_runtime = dict(session.last_runtime)
        session.last_runtime["test_card_answer"] = answer_time

        return {
            "card_id": body.card_id,
            "answer": answer,
            "sources": sources,
            "instruct_warning": instruct_warning,
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Test card answer failed: {exc}") from exc


@router.post("/{document_id}/test-cards/verify")
def verify_cards(document_id: str, body: TestCardsVerifyRequest) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.test_cards:
        raise HTTPException(status_code=400, detail="Generate test cards first.")

    try:
        t0 = time.perf_counter()
        instruct_gen, instruct_warning = get_instruct_generator(body.instruct_model_key)

        cards = [card_from_session_dict(c) for c in session.test_cards]
        user_answers = {a.card_id: a.answer for a in body.answers}
        results, total_score = verify_test_cards(
            cards,
            user_answers,
            generator=instruct_gen,
        )

        session.test_card_results = [score_to_dict(r) for r in results]
        session.last_runtime = dict(session.last_runtime)
        session.last_runtime["test_cards_verify"] = time.perf_counter() - t0

        return {
            "results": session.test_card_results,
            "total_score": total_score,
            "instruct_warning": instruct_warning,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc
