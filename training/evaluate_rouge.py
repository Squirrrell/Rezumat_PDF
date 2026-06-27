"""Evaluate pretrained vs fine-tuned T5-small with ROUGE (CLI wrapper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataset_utils import DEFAULT_DATASET_CONFIG, DEFAULT_MODEL_NAME  # noqa: E402
from service import DEFAULT_RESULTS_JSON, run_rouge_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROUGE evaluation for T5-small models.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to fine-tuned checkpoint (folder with config.json).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--dataset-config", default=DEFAULT_DATASET_CONFIG, choices=["arxiv", "pubmed"]
    )
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=0, help="0 = use val_size")
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--finetuned-only", action="store_true")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_RESULTS_JSON))
    return parser.parse_args()


def _print_scores(label: str, scores: dict[str, float]) -> None:
    print(
        f"{label}: ROUGE-1 {scores['rouge1']:.4f} | "
        f"ROUGE-2 {scores['rouge2']:.4f} | "
        f"ROUGE-L {scores['rougeL']:.4f}"
    )


def main() -> None:
    args = parse_args()

    results = run_rouge_evaluation(
        model_name=args.model_name,
        dataset_config=args.dataset_config,
        checkpoint=args.checkpoint,
        train_size=args.train_size,
        val_size=args.val_size,
        eval_size=args.eval_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        run_baseline=not args.finetuned_only,
        run_finetuned=not args.baseline_only,
        output_json=args.output_json,
    )

    print(f"Device: {results['device']}")
    print(f"Evaluating on {results['eval_size']} examples")
    for label, scores in results["models"].items():
        _print_scores(label, scores)
    if "finetuned" not in results["models"] and not args.baseline_only:
        print("No --checkpoint provided; skipped fine-tuned evaluation.")
    print(f"\nSaved results to {results['output_json']}")


if __name__ == "__main__":
    main()
