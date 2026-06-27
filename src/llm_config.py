"""LLM backend selection (local Hugging Face vs OpenAI API)."""

from __future__ import annotations

import os

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def llm_backend() -> str:
    return os.environ.get("SQUIRRELAI_LLM_BACKEND", "local").strip().lower()


def is_openai_backend() -> bool:
    return llm_backend() == "openai"


def get_openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL


def require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env when SQUIRRELAI_LLM_BACKEND=openai."
        )
    return key


def openai_uses_completion_tokens(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def openai_supports_temperature(model: str) -> bool:
    return not openai_uses_completion_tokens(model)


def effective_max_tokens(base: int, model: str | None = None) -> int:
    """Scale token budget for reasoning models that spend tokens on internal thinking."""
    model_name = (model or get_openai_model()).lower()
    if openai_uses_completion_tokens(model_name):
        return max(base * 2, 800)
    return base


def qa_max_new_tokens(answer_length: str) -> int:
    """Map UI answer_length setting to generation budget."""
    mapping = {"short": 200, "medium": 400, "long": 600}
    return mapping.get(answer_length.strip().lower(), 400)
