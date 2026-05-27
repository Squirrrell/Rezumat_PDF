# Optional T5 Fine-Tuning Experiment (Thesis)

This folder contains an **optional** research experiment for fine-tuning a pretrained summarization model. It is **separate from the main Streamlit app** — you do not need to run any script here to use the PDF summarizer.

## Purpose

This is **not training from scratch**. You start from Google’s pretrained **[google-t5/t5-small](https://huggingface.co/google-t5/t5-small)** (60M parameters) and **fine-tune** it on a small subset of the Hugging Face **[scientific_papers](https://huggingface.co/datasets/scientific_papers)** dataset (article → abstract).

Goals for a bachelor/license thesis:

- Show how domain fine-tuning differs from off-the-shelf pretrained summarization (DistilBART in the app).
- Report **ROUGE-1, ROUGE-2, ROUGE-L** on a held-out validation slice.
- Keep the experiment reproducible on modest hardware (4 GB VRAM, 24 GB RAM).

## Separation from the main app

| Component | Location | Required for app? |
|-----------|----------|-------------------|
| Streamlit summarizer | `app.py`, `src/` | Yes |
| This experiment | `training/` | **No** |

The app continues to use DistilBART, MiniLM, and Qwen/FLAN for structured summaries. Fine-tuned checkpoints are saved under `training/checkpoints/` and are **not** loaded by the app in this version.

## Hardware notes

- **Batch size 1** with **gradient accumulation 16** (effective batch 16) to limit VRAM.
- **Max input length 512**, **max target length 128** tokens.
- **FP16** on CUDA when available; CPU training works but is slow.
- Default subset: **1000 train** + **100 validation** examples (arxiv config).
- Examples are collected via **streaming** (no full 3.6GB arxiv download required).

## Setup

From the project root:

```bash
pip install -r training/requirements-training.txt
```

This installs `datasets` (2.x), `rouge-score`, and training dependencies **without** changing the main `requirements.txt`.

Note: `scientific_papers` requires `datasets>=2.14,<3` and `trust_remote_code=True` when loading (handled in `dataset_utils.py`).

## Dataset

- **Source:** `scientific_papers` on Hugging Face
- **Config:** `arxiv` (alternative: `pubmed`)
- **Input:** full article text, prefixed with `summarize: ` (T5 convention)
- **Target:** abstract (reference summary)
- Rows with empty article or abstract are removed

## Train

```bash
python training/train_t5_small.py --train-size 1000 --val-size 100 --epochs 3
```

Quick dry-run (fewer examples, 1 epoch):

```bash
python training/train_t5_small.py --train-size 50 --val-size 10 --epochs 1 --save-steps 25 --logging-steps 10
```

Checkpoints are written to:

`training/checkpoints/t5-small-scientific/`

A final model is also saved to:

`training/checkpoints/t5-small-scientific/final/`

### Main training hyperparameters

| Parameter | Default |
|-----------|---------|
| Model | google-t5/t5-small |
| Train examples | 1000 |
| Val examples | 100 |
| Batch size | 1 |
| Gradient accumulation | 16 |
| Learning rate | 5e-5 |
| Epochs | 3 |
| Max input tokens | 512 |
| Max target tokens | 128 |

## Evaluate (ROUGE)

Compare **pretrained** T5-small vs your **fine-tuned** checkpoint:

```bash
python training/evaluate_rouge.py --checkpoint training/checkpoints/t5-small-scientific/final --eval-size 100
```

Baseline only:

```bash
python training/evaluate_rouge.py --baseline-only --eval-size 100
```

Results are printed to the console and saved to:

`training/results/rouge_results.json`

### Reading ROUGE scores

Higher F1 is better. Typical thesis table:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Pretrained T5-small | | | |
| Fine-tuned T5-small | | | |

ROUGE measures overlap with reference abstracts; it does not guarantee factual correctness or readability.

## File overview

```
training/
├── dataset_utils.py      # Load subset, tokenize, collator
├── train_t5_small.py     # Fine-tuning script
├── evaluate_rouge.py     # ROUGE evaluation
├── requirements-training.txt
├── README_training.md
├── checkpoints/          # Saved models (gitignored contents)
└── results/              # ROUGE JSON output
```

## Thesis usage

1. Describe this as **transfer learning / fine-tuning**, not training a new architecture from random weights.
2. Contrast with the app’s **DistilBART** (news-trained) and optional **Qwen** structured summaries.
3. Report ROUGE on the same validation slice before and after fine-tuning.
4. Discuss limitations: small subset, single domain (arxiv), ROUGE-only evaluation.

## Limitations

- **Subset only** — not the full scientific_papers corpus.
- **ROUGE ≠ quality** — high overlap can still miss facts or produce dull text.
- **Not integrated in the app** — manual comparison for the thesis chapter.
- **Download size** — first run downloads dataset and model weights from Hugging Face.

## Smoke test

```bash
python training/dataset_utils.py
```
