"""Evaluate pretrained vs fine-tuned T5-small with ROUGE on a held-out subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataset_utils import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    T5_PREFIX,
    load_eval_raw_subset,
)

RESULTS_DIR = TRAINING_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROUGE evaluation for T5-small models.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to fine-tuned checkpoint (folder with config.json).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-config", default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=0, help="0 = use val_size")
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--finetuned-only", action="store_true")
    parser.add_argument("--output-json", type=str, default=str(RESULTS_DIR / "rouge_results.json"))
    return parser.parse_args()


def generate_summaries(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    articles: list[str],
    device: str,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    """Generate abstracts for a list of articles."""
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    preds = generate_summaries(
        model, tokenizer, articles, device, max_input_length, max_new_tokens
    )
    return compute_rouge(preds, references)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    eval_size = args.eval_size if args.eval_size > 0 else args.val_size
    val_raw = load_eval_raw_subset(
        config_name=args.dataset_config,
        train_size=args.train_size,
        val_size=eval_size,
    )
    articles = val_raw["article"]
    references = val_raw["abstract"]
    print(f"Evaluating on {len(articles)} examples")

    results: dict = {
        "dataset_config": args.dataset_config,
        "eval_size": len(articles),
        "models": {},
    }

    run_baseline = not args.finetuned_only
    run_finetuned = not args.baseline_only and bool(args.checkpoint)

    if run_baseline:
        print(f"\nBaseline (pretrained): {args.model_name}")
        results["models"]["pretrained"] = evaluate_model(
            args.model_name,
            articles,
            references,
            device,
            args.max_input_length,
            args.max_new_tokens,
        )
        _print_scores(results["models"]["pretrained"])

    if run_finetuned:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print(f"\nFine-tuned: {ckpt}")
        results["models"]["finetuned"] = evaluate_model(
            str(ckpt),
            articles,
            references,
            device,
            args.max_input_length,
            args.max_new_tokens,
        )
        _print_scores(results["models"]["finetuned"])
    elif not args.baseline_only:
        print("No --checkpoint provided; skipping fine-tuned evaluation.")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


def _print_scores(scores: dict[str, float]) -> None:
    print(
        f"  ROUGE-1: {scores['rouge1']:.4f} | "
        f"ROUGE-2: {scores['rouge2']:.4f} | "
        f"ROUGE-L: {scores['rougeL']:.4f}"
    )


if __name__ == "__main__":
    main()
