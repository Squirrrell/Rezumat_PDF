"""Cached ML model loaders (replaces Streamlit @st.cache_resource)."""

from __future__ import annotations

from functools import lru_cache

from marker.converters.pdf import PdfConverter

from src.instruct_generator import InstructGenerator, load_instruct_model_with_fallback
from src.llm_config import is_openai_backend
from src.marker_converter import create_pdf_converter
from src.model_manager import ModelBundle
from src.summarizer import load_summarizer
from src.vector_store import SentenceTransformer, load_embedding_model


@lru_cache(maxsize=1)
def get_marker_converter() -> PdfConverter:
    return create_pdf_converter()


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return load_embedding_model()


@lru_cache(maxsize=4)
def _get_instruct_generator_cached(cache_key: str) -> tuple[InstructGenerator | None, str | None]:
    _, model_key = cache_key.split(":", 1)
    return load_instruct_model_with_fallback(model_key)


def get_instruct_generator(model_key: str) -> tuple[InstructGenerator | None, str | None]:
    backend = "openai" if is_openai_backend() else "local"
    return _get_instruct_generator_cached(f"{backend}:{model_key}")


@lru_cache(maxsize=1)
def get_summarizer() -> ModelBundle:
    return load_summarizer()
