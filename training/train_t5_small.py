"""Fine-tune google-t5/t5-small on a scientific_papers subset (CLI wrapper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataset_utils import DEFAULT_DATASET_CONFIG, DEFAULT_MODEL_NAME  # noqa: E402
from service import DEFAULT_OUTPUT_DIR, TrainingConfig, run_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune T5-small on scientific_papers (article -> abstract)."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--dataset-config", default=DEFAULT_DATASET_CONFIG, choices=["arxiv", "pubmed"]
    )
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--logging-steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        model_name=args.model_name,
        dataset_config=args.dataset_config,
        train_size=args.train_size,
        val_size=args.val_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
    )

    print(
        f"Training on {config.train_size} examples "
        f"(effective batch size {config.batch_size * config.gradient_accumulation_steps})..."
    )
    result = run_training(config)
    print(f"Device: {result['device']}")
    print(f"Saved final model to {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
