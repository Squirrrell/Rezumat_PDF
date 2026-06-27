"""In-memory document sessions (server-side FAISS + text)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from src.evaluation import DocumentMetrics
from src.section_detector import PaperSections
from src.vector_store import VectorIndex


@dataclass
class DocumentSession:
    document_id: str
    paper_name: str
    pdf_bytes: bytes = b""
    raw_text: str = ""
    markdown: str = ""
    cleaned_text: str = ""
    chunks: list[str] = field(default_factory=list)
    page_count: int = 0
    markdown_ready: bool = False
    sections: PaperSections | None = None
    vector_index: VectorIndex | None = None
    document_metrics: DocumentMetrics | None = None
    summary: str = ""
    summary_metrics: dict | None = None
    summary_source: str | None = None
    comprehensive_summary: str = ""
    comprehensive_summary_metrics: dict | None = None
    comprehensive_summary_source: str | None = None
    test_cards: list[dict[str, Any]] = field(default_factory=list)
    test_card_results: list[dict[str, Any]] | None = None
    qa_answer: str = ""
    qa_sources: list[dict] = field(default_factory=list)
    qa_metrics: dict | None = None
    last_runtime: dict[str, float] = field(default_factory=dict)


_sessions: dict[str, DocumentSession] = {}


def create_session(paper_name: str) -> DocumentSession:
    session = DocumentSession(document_id=str(uuid4()), paper_name=paper_name)
    _sessions[session.document_id] = session
    return session


def get_session(document_id: str) -> DocumentSession:
    session = _sessions.get(document_id)
    if session is None:
        raise KeyError(f"Document session not found: {document_id}")
    return session


def delete_session(document_id: str) -> None:
    _sessions.pop(document_id, None)
