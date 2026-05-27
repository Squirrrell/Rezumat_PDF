"""Section-aware structured summaries via per-field retrieval."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer

from src.instruct_generator import InstructGenerator
from src.vector_store import VectorIndex, search

MISSING_TEXT = "Not clearly specified in the document."
FIELD_TOP_K = 5
# L2 distance on MiniLM embeddings; lower is better. Tune if needed.
WEAK_DISTANCE_THRESHOLD = 1.15
DUPLICATE_RATIO_THRESHOLD = 0.75
RETRY_QUERY_SUFFIX = " specific unique details"

STRUCTURED_SECTIONS = [
    "Title / Topic",
    "Main idea",
    "Problem addressed",
    "Proposed method",
    "Dataset / experiments",
    "Results",
    "Limitations",
    "Conclusion",
    "Key takeaways",
]

FIELD_RETRIEVAL_QUERIES: dict[str, str] = {
    "Title / Topic": "title abstract topic paper subject",
    "Main idea": "main contribution main idea abstract overview introduction",
    "Problem addressed": "problem motivation challenge gap issue limitation existing methods",
    "Proposed method": "method approach proposed framework architecture algorithm system model",
    "Dataset / experiments": "dataset experiment evaluation benchmark setup data training testing",
    "Results": "results findings performance comparison accuracy improvement evaluation",
    "Limitations": "limitations weakness constraints threats future work discussion",
    "Conclusion": "conclusion final remarks summary implications",
    "Key takeaways": "main findings contributions results conclusion important points",
}


@dataclass
class FieldResult:
    """One structured summary field with optional retrieval sources."""

    field_name: str
    text: str
    sources: list[dict] = field(default_factory=list)


@dataclass
class StructuredSummaryResult:
    """Full structured summary output."""

    body: str
    fields: list[FieldResult]


def _dedupe_chunks(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
    seen: set[str] = set()
    unique: list[tuple[str, float]] = []
    for text, score in hits:
        key = text.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append((text, score))
    return unique


def _is_weak_retrieval(hits: list[tuple[str, float]]) -> bool:
    if not hits:
        return True
    best_distance = min(score for _, score in hits)
    return best_distance > WEAK_DISTANCE_THRESHOLD


def _build_context(hits: list[tuple[str, float]]) -> str:
    return "\n\n".join(text for text, _ in hits)


def _hits_to_sources(hits: list[tuple[str, float]]) -> list[dict]:
    return [{"text": text, "score": score} for text, score in hits]


def _normalize_field_text(text: str, field_name: str) -> str:
    text = text.strip()
    if not text:
        return MISSING_TEXT
    lower = text.lower()
    if "not clearly specified" in lower:
        return MISSING_TEXT
    # Strip accidental label prefix from model output
    prefix = f"{field_name}:".lower()
    if lower.startswith(prefix):
        text = text[len(field_name) + 1 :].strip()
    return text or MISSING_TEXT


def _format_key_takeaways(text: str) -> str:
    text = text.strip()
    if not text or text == MISSING_TEXT:
        return MISSING_TEXT
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("-"):
            line = f"- {line.lstrip('•* ')}"
        lines.append(line)
    if not lines:
        parts = [p.strip() for p in text.replace(";", ".").split(".") if p.strip()]
        lines = [f"- {p}" for p in parts[:5]]
    return "\n".join(lines[:5]) if lines else MISSING_TEXT


def _format_body(fields: list[FieldResult]) -> str:
    parts: list[str] = []
    for fr in fields:
        if fr.field_name == "Key takeaways":
            parts.append(f"{fr.field_name}:\n{fr.text}")
        else:
            parts.append(f"{fr.field_name}:\n{fr.text}")
    return "\n\n".join(parts)


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _generate_one_field(
    field_name: str,
    vector_index: VectorIndex,
    embed_model: SentenceTransformer,
    generator: InstructGenerator,
    top_k: int,
    query_override: str | None = None,
) -> FieldResult:
    query = query_override or FIELD_RETRIEVAL_QUERIES[field_name]
    hits = search(vector_index, query, embed_model, top_k=top_k)
    hits = _dedupe_chunks(hits)
    sources = _hits_to_sources(hits)

    if _is_weak_retrieval(hits):
        return FieldResult(field_name=field_name, text=MISSING_TEXT, sources=sources)

    context = _build_context(hits)
    is_key_takeaways = field_name == "Key takeaways"
    raw = generator.generate_field(
        field_name,
        context,
        key_takeaways=is_key_takeaways,
    )

    if is_key_takeaways:
        text = _format_key_takeaways(raw)
    else:
        text = _normalize_field_text(raw, field_name)

    return FieldResult(field_name=field_name, text=text, sources=sources)


def _resolve_duplicates(
    fields: list[FieldResult],
    vector_index: VectorIndex,
    embed_model: SentenceTransformer,
    generator: InstructGenerator,
    top_k: int,
) -> list[FieldResult]:
    resolved: list[FieldResult] = []

    for i, fr in enumerate(fields):
        if fr.text == MISSING_TEXT:
            resolved.append(fr)
            continue

        is_duplicate = False
        for prior in resolved:
            if prior.text == MISSING_TEXT:
                continue
            if _similarity(fr.text, prior.text) > DUPLICATE_RATIO_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            resolved.append(fr)
            continue

        retry_query = FIELD_RETRIEVAL_QUERIES[fr.field_name] + RETRY_QUERY_SUFFIX
        retried = _generate_one_field(
            fr.field_name,
            vector_index,
            embed_model,
            generator,
            top_k,
            query_override=retry_query,
        )

        still_duplicate = False
        for prior in resolved:
            if prior.text == MISSING_TEXT:
                continue
            if _similarity(retried.text, prior.text) > DUPLICATE_RATIO_THRESHOLD:
                still_duplicate = True
                break

        if still_duplicate:
            resolved.append(
                FieldResult(
                    field_name=fr.field_name,
                    text=MISSING_TEXT,
                    sources=fr.sources,
                )
            )
        else:
            resolved.append(retried)

    return resolved


def generate_structured_summary(
    vector_index: VectorIndex,
    embed_model: SentenceTransformer,
    generator: InstructGenerator,
    top_k: int = FIELD_TOP_K,
) -> StructuredSummaryResult:
    """
    Generate a structured paper summary with per-field retrieval and generation.

    Each field uses a different semantic search query so contexts do not repeat.
    """
    fields: list[FieldResult] = []

    for field_name in STRUCTURED_SECTIONS:
        fr = _generate_one_field(
            field_name,
            vector_index,
            embed_model,
            generator,
            top_k,
        )
        fields.append(fr)

    fields = _resolve_duplicates(
        fields,
        vector_index,
        embed_model,
        generator,
        top_k,
    )

    body = _format_body(fields)
    return StructuredSummaryResult(body=body, fields=fields)
