"""PDF upload, ingest, and optional Markdown conversion."""

from __future__ import annotations

import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from backend.model_cache import get_embedder, get_marker_converter
from backend.schemas import DocumentMetadataResponse
from backend.serializers import sections_to_dict
from backend.session_store import create_session, get_session
from src.evaluation import evaluate_document
from src.marker_converter import convert_pdf_to_markdown
from src.pdf_text import extract_text_from_pdf
from src.section_detector import detect_sections
from src.text_utils import chunk_text, normalize_markdown
from src.vector_store import build_index

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=DocumentMetadataResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1200),
    overlap: int = Form(150),
) -> DocumentMetadataResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        t_ingest = time.perf_counter()
        raw_text, page_count = extract_text_from_pdf(file_bytes)
        normalized = normalize_markdown(raw_text)
        if not normalized.strip():
            raise ValueError("PDF text extraction produced no usable content.")
        chunks = chunk_text(normalized, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("PDF was indexed, but chunking produced no chunks.")
        sections = detect_sections(normalized)
        ingest_time = time.perf_counter() - t_ingest

        session = create_session(file.filename)
        session.pdf_bytes = file_bytes
        session.raw_text = raw_text
        session.cleaned_text = normalized
        session.markdown = ""
        session.markdown_ready = False
        session.chunks = chunks
        session.page_count = page_count
        session.sections = sections
        session.summary = ""
        session.summary_metrics = None
        session.summary_source = None
        session.comprehensive_summary = ""
        session.comprehensive_summary_metrics = None
        session.comprehensive_summary_source = None

        t_index = time.perf_counter()
        embed_model = get_embedder()
        session.vector_index = build_index(chunks, embed_model)
        index_time = time.perf_counter() - t_index

        session.document_metrics = evaluate_document(
            normalized,
            page_count=page_count,
            chunk_count=len(chunks),
            runtime_seconds=ingest_time,
        )
        session.test_cards = []
        session.test_card_results = None
        session.qa_answer = ""
        session.qa_sources = []
        session.qa_metrics = None
        session.last_runtime = {
            "pdf_ingest": ingest_time,
            "index_build": index_time,
        }

        return _metadata_response(session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc


@router.post("/{document_id}/convert-markdown", response_model=DocumentMetadataResponse)
def convert_markdown(document_id: str) -> DocumentMetadataResponse:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.pdf_bytes:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    try:
        converter = get_marker_converter()
        t_marker = time.perf_counter()
        markdown, page_count = convert_pdf_to_markdown(session.pdf_bytes, converter=converter)
        marker_time = time.perf_counter() - t_marker

        t_post = time.perf_counter()
        normalized_md = normalize_markdown(markdown)
        if not normalized_md.strip():
            raise ValueError("Marker produced no usable Markdown content.")
        post_time = time.perf_counter() - t_post

        session.raw_text = markdown
        session.markdown = normalized_md
        session.markdown_ready = True
        if page_count > 0:
            session.page_count = page_count

        session.last_runtime = dict(session.last_runtime)
        session.last_runtime["marker_inference"] = marker_time
        session.last_runtime["markdown_post_process"] = post_time

        return _metadata_response(session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Markdown conversion failed: {exc}") from exc


@router.get("/{document_id}", response_model=DocumentMetadataResponse)
def get_document(document_id: str) -> DocumentMetadataResponse:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _metadata_response(session)


@router.get("/{document_id}/markdown")
def get_markdown(document_id: str) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.markdown_ready or not session.markdown:
        raise HTTPException(
            status_code=400,
            detail="Markdown not available. Convert the PDF on the Markdown tab first.",
        )

    return {
        "markdown": session.markdown,
        "page_count": session.page_count,
        "character_count": len(session.markdown),
    }


@router.get("/{document_id}/preview")
def get_preview(document_id: str, limit: int = 2000) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if session.markdown_ready and session.markdown:
        text = session.markdown
    else:
        text = session.cleaned_text

    if not text:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    preview = text[:limit]
    if len(text) > limit:
        preview += "\n\n..."
    return {
        "preview": preview,
        "character_count": len(text),
        "chunk_count": len(session.chunks),
        "page_count": session.page_count,
    }


@router.get("/{document_id}/sections")
def get_sections(document_id: str, include_text: bool = False) -> dict:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.sections or not session.cleaned_text:
        raise HTTPException(status_code=400, detail="Upload a PDF to detect sections.")

    return sections_to_dict(session.sections, include_text=include_text)


@router.get("/{document_id}/export")
def export_markdown(document_id: str) -> PlainTextResponse:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.markdown_ready or not session.markdown:
        raise HTTPException(
            status_code=400,
            detail="Markdown not available. Convert the PDF on the Markdown tab first.",
        )

    filename = session.paper_name
    if filename.lower().endswith(".pdf"):
        filename = filename[:-4] + ".md"
    elif not filename.lower().endswith(".md"):
        filename = f"{filename}.md"

    return PlainTextResponse(
        content=session.markdown,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        media_type="text/markdown; charset=utf-8",
    )


def _metadata_response(session) -> DocumentMetadataResponse:
    sections = session.sections
    return DocumentMetadataResponse(
        document_id=session.document_id,
        paper_name=session.paper_name,
        page_count=session.page_count,
        chunk_count=len(session.chunks),
        character_count=len(session.cleaned_text),
        sections_detected=sections.detected if sections else [],
        sections_coverage_ratio=sections.coverage_ratio if sections else 0.0,
        sections_fallback_body=sections.fallback_body if sections else False,
        has_markdown=session.markdown_ready,
        has_summary=bool(session.summary and session.summary.strip()),
        last_runtime=session.last_runtime,
    )
