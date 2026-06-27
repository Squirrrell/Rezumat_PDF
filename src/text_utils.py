"""Text cleaning and chunking utilities."""

import re


def normalize_markdown(text: str) -> str:
    """Light normalization that preserves Markdown structure."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 4+ blank lines to two newlines.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Trim trailing whitespace per line without touching heading markers.
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def clean_text(text: str) -> str:
    """Remove artifacts and normalize whitespace."""
    if not text:
        return ""

    # Join hyphenated words split across lines
    text = re.sub(r"-\s*\n\s*", "", text)

    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        # Drop isolated page numbers
        if re.fullmatch(r"\d+", line):
            continue
        lines.append(line)

    return "\n".join(lines)


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
    min_chunk_length: int = 50,
) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    Args:
        text: Input text to split.
        chunk_size: Target characters per chunk.
        overlap: Overlap between consecutive chunks.
        min_chunk_length: Skip chunks shorter than this after stripping.

    Returns:
        List of text chunks.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_length:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - overlap

    return chunks
