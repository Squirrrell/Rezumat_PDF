"""OpenAI API backend for instruct tasks and brief summarization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

from src.evaluation import evaluate_summary
from src.instruct_generator import (
    FIELD_PROMPT,
    KEY_TAKEAWAYS_PROMPT,
    MAX_CONTEXT_CHARS,
    QA_INSTRUCT_PROMPT,
    QA_MAX_CONTEXT_CHARS,
)
from src.llm_config import (
    effective_max_tokens,
    get_openai_model,
    openai_supports_temperature,
    openai_uses_completion_tokens,
    require_openai_key,
)
from src.summarizer import LENGTH_PRESETS

CHUNK_SUMMARY_PROMPT = """Summarize the following excerpt from a scientific paper.
Be concise and factual. Do not invent information.

Excerpt:
{text}

Summary:"""

FINAL_SUMMARY_PROMPT = """Combine the following section summaries into one coherent summary
of the scientific paper. Be concise. Do not invent information.

Section summaries:
{text}

Final summary:"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=require_openai_key())
    return _client


def _build_chat_kwargs(model: str, prompt: str, *, max_tokens: int) -> dict:
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if openai_uses_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        if openai_supports_temperature(model):
            kwargs["temperature"] = 0.2
    return kwargs


def _extract_message_content(response) -> str:
    choice = response.choices[0].message.content
    return (choice or "").strip()


def chat_completion(prompt: str, *, max_tokens: int = 400) -> str:
    client = _get_client()
    model = get_openai_model()
    kwargs = _build_chat_kwargs(model, prompt, max_tokens=max_tokens)
    response = client.chat.completions.create(**kwargs)
    return _extract_message_content(response)


@dataclass
class OpenAIInstructGenerator:
    """Drop-in replacement for InstructGenerator when using OpenAI API."""

    model_id: str
    backend: Literal["openai"]
    device: str

    def generate_field(
        self,
        field_name: str,
        context: str,
        *,
        key_takeaways: bool = False,
    ) -> str:
        context = context.strip()[:MAX_CONTEXT_CHARS]
        if not context:
            return ""

        if key_takeaways:
            prompt = KEY_TAKEAWAYS_PROMPT.format(context=context)
        else:
            prompt = FIELD_PROMPT.format(context=context, field_name=field_name)

        return self._generate(prompt)

    def generate_qa_answer(
        self,
        question: str,
        context: str,
        *,
        max_new_tokens: int = 400,
    ) -> str:
        context = context.strip()[:QA_MAX_CONTEXT_CHARS]
        question = question.strip()
        if not context or not question:
            return ""

        prompt = QA_INSTRUCT_PROMPT.format(context=context, question=question)
        budget = effective_max_tokens(max_new_tokens, self.model_id)
        return self._generate(prompt, max_new_tokens=budget)

    def _generate(self, prompt: str, max_new_tokens: int = 250) -> str:
        return chat_completion(prompt, max_tokens=max_new_tokens)


def load_openai_instruct_generator() -> OpenAIInstructGenerator:
    require_openai_key()
    return OpenAIInstructGenerator(
        model_id=get_openai_model(),
        backend="openai",
        device="api",
    )


def summarize_chunks_openai(
    chunks: list[str],
    *,
    length: str = "medium",
    max_chunks: int = 12,
    strategy: str = "hierarchical_openai",
) -> tuple[str, dict]:
    """Hierarchical brief summary via OpenAI (chunk summaries then final pass)."""
    t0 = time.perf_counter()
    preset = LENGTH_PRESETS.get(length, LENGTH_PRESETS["medium"])
    chunk_max = preset["chunk_max_new_tokens"]
    final_max = preset["final_max_new_tokens"]

    selected = [c for c in (chunks or []) if c and c.strip()][: max(1, int(max_chunks))]
    if not selected:
        return "", {
            "strategy": strategy,
            "source_words": 0,
            "summary_words": 0,
            "compression_ratio": 0.0,
            "compression_percent": 0.0,
            "runtime_seconds": 0.0,
            "chunks_used": 0,
            "generated_tokens": 0,
        }

    chunk_summaries: list[str] = []
    for chunk in selected:
        text = chunk.strip()[:MAX_CONTEXT_CHARS]
        prompt = CHUNK_SUMMARY_PROMPT.format(text=text)
        summary = chat_completion(prompt, max_tokens=chunk_max)
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        runtime = time.perf_counter() - t0
        return "", {
            "strategy": strategy,
            "source_words": 0,
            "summary_words": 0,
            "compression_ratio": 0.0,
            "compression_percent": 0.0,
            "runtime_seconds": runtime,
            "chunks_used": len(selected),
            "generated_tokens": 0,
        }

    combined = "\n\n".join(chunk_summaries)
    final_prompt = FINAL_SUMMARY_PROMPT.format(text=combined[:MAX_CONTEXT_CHARS])
    final = chat_completion(final_prompt, max_tokens=final_max)

    runtime = time.perf_counter() - t0
    source_text = "\n\n".join(selected)
    metrics = evaluate_summary(source_text, final, runtime, strategy=strategy)
    metrics_dict = {
        "strategy": metrics.strategy,
        "source_words": metrics.source_words,
        "summary_words": metrics.summary_words,
        "compression_ratio": metrics.compression_ratio,
        "compression_percent": metrics.compression_percent,
        "runtime_seconds": metrics.runtime_seconds,
        "chunks_used": len(selected),
        "generated_tokens": 0,
    }
    return final, metrics_dict
