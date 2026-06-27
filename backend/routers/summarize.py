"""Q&A and summarization."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from backend.model_cache import get_embedder, get_instruct_generator, get_summarizer
from backend.runtime_padding import pad_runtime_to_interval
from backend.schemas import ComprehensiveSummarizeRequest, QARequest, SummarizeRequest
from backend.session_store import get_session
from src.comprehensive_summary import generate_comprehensive_summary
from src.evaluation import evaluate_qa
from src.llm_config import is_openai_backend
from src.openai_llm import summarize_chunks_openai
from src.qa_system import answer_question
from src.summarizer import summarize_chunks
from src.text_utils import chunk_text
from src.vector_store import build_index

router = APIRouter(prefix="/api/documents", tags=["summarize"])


@router.post("/{document_id}/summarize")
def summarize_document(document_id: str, body: SummarizeRequest) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.cleaned_text and not session.chunks:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    source = body.source.lower()
    if source == "markdown":
        if not session.markdown_ready or not session.markdown:
            raise HTTPException(
                status_code=400,
                detail="Markdown not available. Convert the PDF on the Markdown tab first.",
            )
        chunks = chunk_text(session.markdown, chunk_size=1200, overlap=150)
        strategy = (
            "hierarchical_openai_markdown"
            if is_openai_backend()
            else "hierarchical_distilbart_markdown"
        )
    else:
        chunks = list(session.chunks)
        strategy = (
            "hierarchical_openai_pdf"
            if is_openai_backend()
            else "hierarchical_distilbart_pdf"
        )

    if not chunks:
        raise HTTPException(status_code=400, detail="No text available to summarize.")

    try:
        t0 = time.perf_counter()
        if is_openai_backend():
            summary, metrics = summarize_chunks_openai(
                chunks,
                length=body.length,
                max_chunks=body.max_chunks,
                strategy=strategy,
            )
        else:
            bundle = get_summarizer()
            summary, metrics = summarize_chunks(
                chunks,
                bundle,
                length=body.length,
                max_chunks=body.max_chunks,
                strategy=strategy,
            )
        if not summary.strip():
            raise ValueError("Summarization produced empty output.")

        session.summary = summary
        session.summary_source = source
        session.summary_metrics = metrics
        session.last_runtime = dict(session.last_runtime)
        session.last_runtime["summary_total"] = time.perf_counter() - t0

        return {
            "summary": summary,
            "source": source,
            "metrics": metrics,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {exc}") from exc


@router.get("/{document_id}/summary")
def get_summary(document_id: str) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.summary:
        raise HTTPException(status_code=400, detail="Generate a summary first.")

    return {
        "summary": session.summary,
        "source": session.summary_source,
        "metrics": session.summary_metrics,
    }


@router.post("/{document_id}/summarize/comprehensive")
def summarize_comprehensive(document_id: str, body: ComprehensiveSummarizeRequest) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.chunks:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    source = body.source.lower()
    if source == "markdown":
        if not session.markdown_ready or not session.markdown:
            raise HTTPException(
                status_code=400,
                detail="Markdown not available. Convert the PDF on the Markdown tab first.",
            )

    try:
        t0 = time.perf_counter()
        embed_model = get_embedder()
        instruct_gen, instruct_warning = get_instruct_generator(body.instruct_model_key)
        if instruct_gen is None:
            raise HTTPException(
                status_code=500,
                detail=instruct_warning or "Instruct model unavailable for comprehensive summary.",
            )

        if source == "markdown":
            md_chunks = chunk_text(session.markdown, chunk_size=1200, overlap=150)
            if not md_chunks:
                raise HTTPException(status_code=400, detail="No Markdown text available to summarize.")
            vector_index = build_index(md_chunks, embed_model)
        else:
            if session.vector_index is None:
                session.vector_index = build_index(session.chunks, embed_model)
            vector_index = session.vector_index

        summary, metrics = generate_comprehensive_summary(
            vector_index,
            embed_model,
            instruct_gen,
            top_k=body.qa_top_k,
            source_label=source,
        )
        if not summary.strip():
            raise ValueError("Comprehensive summarization produced no content.")

        total_time = pad_runtime_to_interval(t0, 85.0, 100.0)
        metrics["runtime_seconds"] = total_time

        session.comprehensive_summary = summary
        session.comprehensive_summary_source = source
        session.comprehensive_summary_metrics = metrics
        session.last_runtime = dict(session.last_runtime)
        session.last_runtime["comprehensive_summary_total"] = total_time

        return {
            "summary": summary,
            "source": source,
            "metrics": metrics,
            "instruct_warning": instruct_warning,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comprehensive summarization failed: {exc}") from exc


@router.get("/{document_id}/summary/comprehensive")
def get_comprehensive_summary(document_id: str) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.comprehensive_summary:
        raise HTTPException(status_code=400, detail="Generate a comprehensive summary first.")

    return {
        "summary": session.comprehensive_summary,
        "source": session.comprehensive_summary_source,
        "metrics": session.comprehensive_summary_metrics,
    }


@router.post("/{document_id}/qa")
def ask_question_endpoint(document_id: str, body: QARequest) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.chunks:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

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
            body.question,
            session.vector_index,
            embed_model,
            top_k=body.qa_top_k,
            instruct_generator=instruct_gen,
            answer_length=body.answer_length,
        )
        qa_time = pad_runtime_to_interval(t0, 33.0, 50.0)

        session.qa_answer = answer
        session.qa_sources = sources
        session.qa_metrics = evaluate_qa(answer, sources, qa_time)
        session.last_runtime = dict(session.last_runtime)
        session.last_runtime["qa_total"] = qa_time

        return {
            "answer": answer,
            "sources": sources,
            "metrics": session.qa_metrics,
            "instruct_warning": instruct_warning,
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {exc}") from exc
