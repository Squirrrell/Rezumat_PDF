"""Heuristic section detection for research papers."""

import re
from dataclasses import dataclass, field

from src.text_utils import chunk_text

# Maps canonical section keys to heading patterns (case-insensitive).
SECTION_PATTERNS: dict[str, list[str]] = {
    "abstract": [
        r"^abstract\s*$",
        r"^a\s+b\s+s\s+t\s+r\s+a\s+c\s+t\s*$",
    ],
    "introduction": [
        r"^introduction\s*$",
        r"^1\.?\s*introduction\s*$",
        r"^i\.?\s*introduction\s*$",
    ],
    "methodology": [
        r"^methodology\s*$",
        r"^methods?\s*$",
        r"^materials\s+and\s+methods\s*$",
        r"^experimental\s+(setup|methodology|methods)\s*$",
        r"^2\.?\s*(methodology|methods?|materials\s+and\s+methods)\s*$",
    ],
    "results": [
        r"^results?\s*$",
        r"^findings\s*$",
        r"^discussion\s*$",
        r"^results?\s+and\s+discussion\s*$",
        r"^3\.?\s*(results?|findings|discussion)\s*$",
    ],
    "conclusion": [
        r"^conclusions?\s*$",
        r"^summary\s*$",
        r"^concluding\s+remarks\s*$",
        r"^4\.?\s*conclusions?\s*$",
        r"^5\.?\s*conclusions?\s*$",
    ],
}

# Order used when multiple sections are requested.
SECTION_ORDER = ["abstract", "introduction", "methodology", "results", "conclusion"]


@dataclass
class PaperSections:
    """Detected paper sections."""

    sections: dict[str, str] = field(default_factory=dict)
    detected: list[str] = field(default_factory=list)
    coverage_ratio: float = 0.0
    fallback_body: bool = False


def _match_section(line: str) -> str | None:
    """Return canonical section key if line looks like a heading."""
    stripped = re.sub(r"^#+\s*", "", line.strip())
    if not stripped or len(stripped) > 80:
        return None

    # ALL CAPS short heading (e.g. INTRODUCTION)
    if stripped.isupper() and len(stripped) < 60:
        caps_key = stripped.lower().replace("-", " ").strip()
        for key in SECTION_PATTERNS:
            if caps_key == key or caps_key.startswith(key):
                return key

    lower = stripped.lower()
    for key, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, lower, re.IGNORECASE):
                return key

    return None


def detect_sections(text: str) -> PaperSections:
    """
    Split cleaned paper text into sections using heading heuristics.

    If no headings are found, stores full text under 'body' with fallback_body=True.
    """
    if not text or not text.strip():
        return PaperSections(fallback_body=True)

    lines = text.splitlines()
    headings: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        key = _match_section(line)
        if key:
            # Avoid duplicate consecutive same section
            if headings and headings[-1][1] == key:
                continue
            headings.append((i, key))

    if not headings:
        return PaperSections(
            sections={"body": text.strip()},
            detected=["body"],
            coverage_ratio=1.0,
            fallback_body=True,
        )

    sections: dict[str, str] = {}
    detected: list[str] = []

    for idx, (line_no, key) in enumerate(headings):
        start = line_no + 1
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body_lines = lines[start:end]
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        if key in sections:
            sections[key] = sections[key] + "\n\n" + body
        else:
            sections[key] = body
            detected.append(key)

    total_chars = len(text)
    covered = sum(len(v) for v in sections.values())
    coverage = covered / total_chars if total_chars else 0.0

    return PaperSections(
        sections=sections,
        detected=detected,
        coverage_ratio=min(coverage, 1.0),
        fallback_body=False,
    )


def get_section_text(sections: PaperSections, *keys: str) -> str:
    """Concatenate text from named sections in canonical order."""
    parts: list[str] = []
    for key in keys:
        text = sections.sections.get(key, "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def sections_to_chunks(
    sections: PaperSections,
    chunk_size: int = 1200,
    overlap: int = 150,
    *section_keys: str,
) -> list[str]:
    """Chunk text from selected sections."""
    if section_keys:
        combined = get_section_text(sections, *section_keys)
    else:
        combined = "\n\n".join(sections.sections.values())
    return chunk_text(combined, chunk_size=chunk_size, overlap=overlap)


def intro_conclusion_fallback(text: str, fraction: float = 0.2) -> str:
    """
    Fallback when intro/conclusion sections are missing: first and last fraction of doc.
    """
    text = text.strip()
    if not text:
        return ""
    n = max(len(text) // 5, 500)
    n = min(n, len(text) // 2)
    head = text[:n]
    tail = text[-n:]
    return head + "\n\n" + tail
