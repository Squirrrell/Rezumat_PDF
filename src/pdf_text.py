"""Fast PDF text extraction for ingest (no Marker)."""

from __future__ import annotations

import io

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract embedded text from a digital PDF.

    Returns:
        Tuple of (full text, page count).

    Raises:
        ValueError: If the PDF is empty or yields no text.
    """
    if not file_bytes:
        raise ValueError("No PDF data provided.")

    reader = PdfReader(io.BytesIO(file_bytes))
    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError("PDF has no pages.")

    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())

    full_text = "\n\n".join(parts).strip()
    if not full_text:
        raise ValueError(
            "No extractable text found. This PDF may be scanned; use Markdown conversion or OCR."
        )

    return full_text, page_count
