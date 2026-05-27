"""Question answering via retrieval and summarization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.summarizer import GenerationSettings, summarize_text
from src.vector_store import VectorIndex, search

if TYPE_CHECKING:
    from src.instruct_generator import InstructGenerator

QA_PROMPT = """Answer the question using ONLY the context below.
Do not add information that is not supported by the context.
If the context does not contain enough information, say:
"The available PDF text does not provide enough detail to answer fully."

Question: {question}

Context:
{context}
"""

ANSWER_PRESETS: dict[str, dict[str, int]] = {
    "short": {"max_length": 180, "min_length": 60},
    "medium": {"max_length": 320, "min_length": 100},
    "long": {"max_length": 480, "min_length": 140},
}

WEAK_RETRIEVAL_DISTANCE = 1.15
MIN_CONTEXT_CHARS = 500


def get_answer_preset(answer_length: str) -> GenerationSettings:
    """Build generation settings for Q&A from length preset."""
    preset = ANSWER_PRESETS.get(answer_length.lower(), ANSWER_PRESETS["medium"])
    return GenerationSettings(
        max_length=preset["max_length"],
        min_length=preset["min_length"],
        max_new_tokens=preset["max_length"],
    )


def _limitation_notice(context: str, sources: list[tuple[str, float]]) -> str:
    """Return a prefix notice when context is thin or retrieval is weak."""
    weak = all(score > WEAK_RETRIEVAL_DISTANCE for _, score in sources) if sources else True
    thin = len(context.strip()) < MIN_CONTEXT_CHARS
    if not weak and not thin:
        return ""
    return (
        "**Note:** The answer is based only on the retrieved PDF passages below. "
        "The available text may not contain enough detail for a complete answer.\n\n"
    )


def answer_question(
    question: str,
    vector_index: VectorIndex,
    embed_model: SentenceTransformer,
    tokenizer: AutoTokenizer,
    summarizer_model: AutoModelForSeq2SeqLM,
    device: str,
    top_k: int = 5,
    settings: GenerationSettings | None = None,
    answer_length: str = "medium",
    use_instruct_qa: bool = False,
    instruct_generator: InstructGenerator | None = None,
) -> tuple[str, list[dict]]:
    """
    Answer a question by retrieving relevant chunks and summarizing them.

    Returns:
        Tuple of (answer text, list of source dicts with 'text' and 'score').
    """
    question = question.strip()
    if not question:
        return "", []

    hits = search(vector_index, question, embed_model, top_k=top_k)
    if not hits:
        return "No relevant passages found in the document.", []

    context_parts = [text for text, _ in hits]
    context = "\n\n".join(context_parts)

    if use_instruct_qa and instruct_generator is not None:
        answer = instruct_generator.generate_qa_answer(question, context)
    else:
        if settings is None:
            settings = get_answer_preset(answer_length)
        prompt = QA_PROMPT.format(question=question, context=context)
        answer = summarize_text(
            prompt,
            tokenizer,
            summarizer_model,
            device,
            settings=settings,
        )

    notice = _limitation_notice(context, hits)
    if notice:
        answer = notice + answer

    sources = [{"text": text, "score": score} for text, score in hits]
    return answer, sources
