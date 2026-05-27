"""Instruction-tuned models for structured summaries and Q&A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.summarizer import get_device

QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FLAN_MODEL = "google/flan-t5-small"

MAX_CONTEXT_CHARS = 6000
MAX_NEW_TOKENS = 250

FIELD_PROMPT = """You are analyzing a scientific paper.
Answer ONLY using the provided context.
Do not invent information.
If the context does not contain enough information, say:
"Not clearly specified in the document."

Context:
{context}

Task:
Write the "{field_name}" section of the structured summary.

Rules:
- Be specific.
- Do not repeat the same generic sentence.
- Use 2-3 sentences maximum.
- Focus only on this field.
- Do not mention information that is not in the context.

Answer:"""

KEY_TAKEAWAYS_PROMPT = """You are analyzing a scientific paper.
Answer ONLY using the provided context.
Write 3-5 concrete key takeaways as bullet points.
Do not invent information.
Avoid repeating the same point.

Context:
{context}

Key takeaways:"""

QA_INSTRUCT_PROMPT = """You are analyzing a scientific paper.
Answer ONLY using the provided context.
Do not invent information.
If the context does not contain enough information, say:
"The available PDF text does not provide enough detail to answer fully."

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class InstructGenerator:
    """Wrapper for Qwen (causal) or FLAN-T5 (seq2seq) field generation."""

    model_id: str
    backend: Literal["qwen", "flan"]
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM | AutoModelForSeq2SeqLM
    device: str

    def generate_field(
        self,
        field_name: str,
        context: str,
        *,
        key_takeaways: bool = False,
    ) -> str:
        """Generate one structured field from context."""
        context = context.strip()[:MAX_CONTEXT_CHARS]
        if not context:
            return ""

        if key_takeaways:
            prompt = KEY_TAKEAWAYS_PROMPT.format(context=context)
        else:
            prompt = FIELD_PROMPT.format(context=context, field_name=field_name)

        return self._generate(prompt)

    def generate_qa_answer(self, question: str, context: str) -> str:
        """Generate a Q&A answer from retrieved context."""
        context = context.strip()[:MAX_CONTEXT_CHARS]
        question = question.strip()
        if not context or not question:
            return ""

        prompt = QA_INSTRUCT_PROMPT.format(context=context, question=question)
        return self._generate(prompt, max_new_tokens=320)

    def _generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        if self.backend == "qwen":
            return self._generate_qwen(prompt, max_new_tokens)
        return self._generate_flan(prompt, max_new_tokens)

    def _generate_qwen(self, prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = prompt
        else:
            text = prompt

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_flan(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                repetition_penalty=1.2,
                early_stopping=True,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def load_instruct_model(model_choice: str) -> InstructGenerator:
    """
    Load Qwen or FLAN for structured generation.

    model_choice: "qwen" or "flan"
    """
    choice = model_choice.lower()
    if "flan" in choice:
        return _load_flan()
    try:
        return _load_qwen()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {QWEN_MODEL}. Try flan-t5-small fallback. Details: {exc}"
        ) from exc


def load_instruct_model_with_fallback(model_choice: str) -> tuple[InstructGenerator, str | None]:
    """
    Load preferred model; fall back to FLAN on failure.

    Returns:
        (generator, warning_message or None)
    """
    if "flan" in model_choice.lower():
        return _load_flan(), None

    try:
        return _load_qwen(), None
    except Exception as exc:
        warning = (
            f"Could not load {QWEN_MODEL} ({exc}). "
            f"Using fallback {FLAN_MODEL}."
        )
        return _load_flan(), warning


def _load_qwen() -> InstructGenerator:
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    return InstructGenerator(
        model_id=QWEN_MODEL,
        backend="qwen",
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def _load_flan() -> InstructGenerator:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL)
    model.to(device)
    model.eval()

    return InstructGenerator(
        model_id=FLAN_MODEL,
        backend="flan",
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
