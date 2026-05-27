"""Fine-tune google-t5/t5-small on a scientific_papers subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataset_utils import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    get_data_collator,
    load_scientific_papers_subset,
)

DEFAULT_OUTPUT_DIR = TRAINING_DIR / "checkpoints" / "t5-small-scientific"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune T5-small on scientific_papers (article -> abstract)."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-config", default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")

    train_dataset, val_dataset, tokenizer = load_scientific_papers_subset(
        config_name=args.dataset_config,
        train_size=args.train_size,
        val_size=args.val_size,
        model_name=args.model_name,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    data_collator = get_data_collator(tokenizer, model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=use_cuda,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer_kwargs: dict = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
    }
    # transformers 5.x renamed tokenizer -> processing_class
    try:
        trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=tokenizer)

    print(
        f"Training on {len(train_dataset)} examples "
        f"(effective batch size {args.batch_size * args.gradient_accumulation_steps})..."
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Saved final model to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
