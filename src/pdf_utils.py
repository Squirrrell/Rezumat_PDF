"""PDF text extraction using PyMuPDF."""

import fitz


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract text from a PDF file.

    Returns:
        Tuple of (extracted text, page count).

    Raises:
        ValueError: If the PDF is empty or has no extractable text.
    """
    if not file_bytes:
        raise ValueError("No PDF data provided.")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        page_count = len(doc)
        if page_count == 0:
            raise ValueError(
                "No extractable text found. This may be a scanned PDF without OCR."
            )

        parts: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text:
                parts.append(text)

        full_text = "\n".join(parts).strip()
        if not full_text:
            raise ValueError(
                "No extractable text found. This may be a scanned PDF without OCR."
            )

        return full_text, page_count
    finally:
        doc.close()
