"""Aggregated metrics for thesis tab."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas import MetricsResponse
from backend.serializers import document_metrics_to_dict, sections_to_dict
from backend.session_store import get_session

router = APIRouter(prefix="/api/documents", tags=["metrics"])


@router.get("/{document_id}/metrics", response_model=MetricsResponse)
def get_metrics(document_id: str) -> MetricsResponse:
    try:
        session = get_session(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return MetricsResponse(
        document_metrics=(
            document_metrics_to_dict(session.document_metrics)
            if session.document_metrics
            else None
        ),
        summary_metrics=session.summary_metrics,
        qa_metrics=session.qa_metrics,
        sections=(
            sections_to_dict(session.sections)
            if session.sections and session.cleaned_text
            else None
        ),
        last_runtime=session.last_runtime,
    )
