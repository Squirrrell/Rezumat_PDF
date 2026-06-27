"""Model loading and inference helpers for Model Comparison mode (T5-only).

This module is intentionally independent of the existing DistilBART summarizer so
we can compare pretrained vs locally fine-tuned T5 checkpoints fairly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

T5_PREFIX = "summarize: "
DEFAULT_PRETRAINED_MODEL = "google-t5/t5-small"


@dataclass(frozen=True)
class ModelBundle:
    tokenizer: Any
    model: Any
    device: str
    source: str


def _prefer_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_tokenizer(model_id_or_path: str):
    return AutoTokenizer.from_pretrained(model_id_or_path)


def _load_model(model_id_or_path: str, device: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_path)
    model.to(device)
    model.eval()
    model.generation_config.max_length = None
    return model


def _load_with_cpu_fallback(model_id_or_path: str, source_label: str) -> ModelBundle:
    device = _prefer_device()
    tokenizer = _load_tokenizer(model_id_or_path)
    try:
        model = _load_model(model_id_or_path, device=device)
        return ModelBundle(tokenizer=tokenizer, model=model, device=device, source=source_label)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        torch.cuda.empty_cache()
        cpu_model = _load_model(model_id_or_path, device="cpu")
        return ModelBundle(tokenizer=tokenizer, model=cpu_model, device="cpu", source=source_label)


def load_pretrained_model(model_name: str = DEFAULT_PRETRAINED_MODEL) -> ModelBundle:
    """Load a pretrained summarization model (T5 family)."""
    return _load_with_cpu_fallback(model_name, source_label=model_name)


def load_finetuned_model(model_path: str) -> ModelBundle:
    """Load a locally fine-tuned model from a Hugging Face checkpoint folder."""
    p = Path(model_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Fine-tuned model folder not found: {model_path}")
    # Minimal sanity check for HF-style checkpoints.
    if not (p / "config.json").exists():
        raise ValueError(
            f"Folder does not look like a HF checkpoint (missing config.json): {model_path}"
        )
    return _load_with_cpu_fallback(str(p), source_label=str(p))


def generate_summary_with_model(
    text: str,
    tokenizer,
    model,
    device: str,
    generation_config: dict[str, Any] | None = None,
    *,
    max_input_length: int = 512,
) -> tuple[str, int]:
    """
    Generate a single summary for a given text.

    Returns:
        (summary_text, approx_generated_tokens)
    """
    if not text or not text.strip():
        return "", 0

    generation_config = dict(generation_config or {})
    max_new_tokens = int(generation_config.pop("max_new_tokens", 128))
    num_beams = int(generation_config.pop("num_beams", 4))
    early_stopping = bool(generation_config.pop("early_stopping", True))

    model.generation_config.max_length = None
    gen_config = GenerationConfig(
        max_length=None,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=early_stopping,
    )
    for key, value in generation_config.items():
        if hasattr(gen_config, key):
            setattr(gen_config, key, value)

    inputs = tokenizer(
        T5_PREFIX + text.strip(),
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=gen_config)

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # Approx tokens: output length (minus any special tokens already stripped by decode).
    approx_tokens = int(output_ids.shape[-1])
    return summary, approx_tokens


def hierarchical_summarize_for_model(
    chunks: list[str],
    *,
    tokenizer,
    model,
    device: str,
    max_chunks: int = 12,
    generation_config_chunk: dict[str, Any] | None = None,
    generation_config_final: dict[str, Any] | None = None,
    max_input_length: int = 512,
) -> tuple[str, dict[str, float | int]]:
    """
    Two-pass hierarchical summarization (chunk summaries -> final summary).

    Returns:
        (final_summary, stats)
    """
    t0 = time.perf_counter()

    selected = [c for c in (chunks or []) if c and c.strip()][: max(1, int(max_chunks))]
    if not selected:
        return "", {"runtime_seconds": 0.0, "generated_tokens": 0, "chunks_used": 0}

    chunk_summaries: list[str] = []
    generated_tokens = 0

    for chunk in selected:
        s, tok = generate_summary_with_model(
            chunk,
            tokenizer,
            model,
            device,
            generation_config=generation_config_chunk,
            max_input_length=max_input_length,
        )
        if s:
            chunk_summaries.append(s)
        generated_tokens += tok

    if not chunk_summaries:
        return "", {"runtime_seconds": 0.0, "generated_tokens": 0, "chunks_used": len(selected)}

    combined = "\n\n".join(chunk_summaries)
    final, tok_final = generate_summary_with_model(
        combined,
        tokenizer,
        model,
        device,
        generation_config=generation_config_final,
        max_input_length=max_input_length,
    )
    generated_tokens += tok_final

    runtime = time.perf_counter() - t0
    stats: dict[str, float | int] = {
        "runtime_seconds": float(runtime),
        "generated_tokens": int(generated_tokens),
        "chunks_used": int(len(selected)),
    }
    return final, stats

