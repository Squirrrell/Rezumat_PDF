"""Embedding and FAISS vector search."""

from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.summarizer import get_device

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class VectorIndex:
    """FAISS index with associated chunk texts."""

    index: faiss.IndexFlatL2
    chunks: list[str]


def load_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer embedding model."""
    device = get_device()
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}'. "
            f"Details: {exc}"
        ) from exc


def build_index(chunks: list[str], model: SentenceTransformer) -> VectorIndex:
    """Encode chunks and build a FAISS L2 index."""
    if not chunks:
        raise ValueError("Cannot build index: no chunks provided.")

    embeddings = model.encode(
        chunks,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return VectorIndex(index=index, chunks=chunks)


def search(
    vector_index: VectorIndex,
    query: str,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Search for the most similar chunks to a query.

    Returns:
        List of (chunk_text, distance) tuples, sorted by relevance.
    """
    if not query.strip():
        return []

    top_k = min(top_k, len(vector_index.chunks))
    if top_k == 0:
        return []

    query_embedding = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    distances, indices = vector_index.index.search(query_embedding, top_k)

    results: list[tuple[str, float]] = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        results.append((vector_index.chunks[idx], float(dist)))

    return results
