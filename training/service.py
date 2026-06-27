"""Reusable training and ROUGE-evaluation logic.

Shared by the CLI scripts (`train_t5_small.py`, `evaluate_rouge.py`) and the
backend API (`backend/routers/training.py`). Heavy ML dependencies (`datasets`,
`rouge_score`, the Hugging Face `Trainer`) are imported lazily inside the
functions that need them so this module can be imported cheaply by the backend
even when the optional training extras are not installed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataset_utils import (  # noqa: E402
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    T5_PREFIX,
    get_data_collator,
    load_eval_raw_subset,
    load_scientific_papers_subset,
)

DEFAULT_OUTPUT_DIR = TRAINING_DIR / "checkpoints" / "t5-small-scientific"
RESULTS_DIR = TRAINING_DIR / "results"
DEFAULT_RESULTS_JSON = RESULTS_DIR / "rouge_results.json"

ProgressFn = Callable[[dict], None]
CancelFn = Callable[[], bool]


@dataclass
class TrainingConfig:
    """Hyperparameters for a single fine-tuning run."""

    model_name: str = DEFAULT_MODEL_NAME
    dataset_config: str = DEFAULT_DATASET_CONFIG
    train_size: int = 1000
    val_size: int = 100
    max_input_length: int = MAX_INPUT_LENGTH
    max_target_length: int = MAX_TARGET_LENGTH
    epochs: int = 5
    learning_rate: float = 5e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    logging_steps: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


def run_training(
    config: TrainingConfig,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> dict:
    """Fine-tune T5-small and save the final model.

    on_progress receives a dict with epoch/step/log info after each log event.
    should_cancel is polled during training; returning True stops gracefully.
    """
    from transformers import (
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    use_cuda = torch.cuda.is_available()

    train_dataset, val_dataset, tokenizer = load_scientific_papers_subset(
        config_name=config.dataset_config,
        train_size=config.train_size,
        val_size=config.val_size,
        model_name=config.model_name,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    data_collator = get_data_collator(tokenizer, model)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        fp16=use_cuda,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=config.logging_steps,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    class _ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if on_progress and logs is not None:
                on_progress(
                    {
                        "epoch": state.epoch,
                        "step": state.global_step,
                        "logs": dict(logs),
                        "log_history": list(state.log_history),
                    }
                )
            if should_cancel and should_cancel():
                control.should_training_stop = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if should_cancel and should_cancel():
                control.should_training_stop = True
            return control

    callbacks = [_ProgressCallback()] if (on_progress or should_cancel) else []

    trainer_kwargs: dict = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "callbacks": callbacks,
    }
    # transformers 5.x renamed tokenizer -> processing_class
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=tokenizer)

    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    return {
        "checkpoint_path": str(final_dir),
        "log_history": list(trainer.state.log_history),
        "device": "cuda" if use_cuda else "cpu",
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
    }


def generate_summaries(
    model,
    tokenizer,
    articles: list[str],
    device: str,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    """Generate abstracts for a list of articles."""
    from tqdm import tqdm

    predictions: list[str] = []
    model.eval()

    for article in tqdm(articles, desc="Generating", leave=False):
        text = T5_PREFIX + (article or "")
        inputs = tokenizer(
            text,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred.strip())

    return predictions


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Average ROUGE-1, ROUGE-2, ROUGE-L F1 over the eval set."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        totals["rouge1"] += scores["rouge1"].fmeasure
        totals["rouge2"] += scores["rouge2"].fmeasure
        totals["rougeL"] += scores["rougeL"].fmeasure

    if n == 0:
        return {k: 0.0 for k in totals}

    return {k: v / n for k, v in totals.items()}


def evaluate_model(
    model_path: str,
    articles: list[str],
    references: list[str],
    device: str,
    max_input_length: int,
    max_new_tokens: int,
) -> dict[str, float]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    preds = generate_summaries(
        model, tokenizer, articles, device, max_input_length, max_new_tokens
    )
    return compute_rouge(preds, references)


def run_rouge_evaluation(
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    checkpoint: str = "",
    train_size: int = 1000,
    val_size: int = 100,
    eval_size: int = 0,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_new_tokens: int = MAX_TARGET_LENGTH,
    run_baseline: bool = True,
    run_finetuned: bool = True,
    output_json: str | None = None,
) -> dict:
    """Evaluate pretrained and/or fine-tuned T5 with ROUGE; persist JSON."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_n = eval_size if eval_size > 0 else val_size
    val_raw = load_eval_raw_subset(
        config_name=dataset_config,
        train_size=train_size,
        val_size=eval_n,
    )
    articles = val_raw["article"]
    references = val_raw["abstract"]

    results: dict = {
        "dataset_config": dataset_config,
        "eval_size": len(articles),
        "device": device,
        "models": {},
    }

    if run_baseline:
        results["models"]["pretrained"] = evaluate_model(
            model_name,
            articles,
            references,
            device,
            max_input_length,
            max_new_tokens,
        )

    if run_finetuned and checkpoint:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        results["models"]["finetuned"] = evaluate_model(
            str(ckpt),
            articles,
            references,
            device,
            max_input_length,
            max_new_tokens,
        )

    output_path = Path(output_json) if output_json else DEFAULT_RESULTS_JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    results["output_json"] = str(output_path)

    return results


def list_checkpoints() -> list[dict]:
    """List saved model folders under the default checkpoint directory."""
    base = DEFAULT_OUTPUT_DIR
    checkpoints: list[dict] = []
    if base.exists():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                checkpoints.append({"name": child.name, "path": str(child)})
    return checkpoints


def find_final_checkpoint() -> str | None:
    """Return the path to the saved final model, if it exists."""
    final_dir = DEFAULT_OUTPUT_DIR / "final"
    if (final_dir / "config.json").exists():
        return str(final_dir)
    return None


def load_last_results() -> dict | None:
    """Load the most recent ROUGE results JSON, if present."""
    if DEFAULT_RESULTS_JSON.exists():
        with open(DEFAULT_RESULTS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return None
