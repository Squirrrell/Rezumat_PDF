# T5 Fine-Tuning Experiment (Thesis)

This folder contains a research experiment for fine-tuning a pretrained summarization model. You can run it two ways: from the command line with the scripts here, or from the **Training** tab in the app (which calls the same code through the backend API). The fine-tuned model is still **not** used by the PDF summarizer itself — it is kept separate for the thesis comparison.

## Purpose

This is **not training from scratch**. You start from Google’s pretrained **[google-t5/t5-small](https://huggingface.co/google-t5/t5-small)** (60M parameters) and **fine-tune** it on a small subset of the Hugging Face **[scientific_papers](https://huggingface.co/datasets/scientific_papers)** dataset (article → abstract).

Goals for a bachelor/license thesis:

- Show how domain fine-tuning differs from off-the-shelf pretrained summarization (DistilBART in the app).
- Report **ROUGE-1, ROUGE-2, ROUGE-L** on a held-out validation slice.
- Keep the experiment reproducible on modest hardware (4 GB VRAM, 24 GB RAM).

## Separation from the main app

| Component | Location | Required for app? |
|-----------|----------|-------------------|
| Main summarizer API | `backend/`, `frontend/`, `src/` | Yes |
| This experiment | `training/` | **No** |

The app continues to use DistilBART, MiniLM, and Qwen/FLAN for structured summaries. Fine-tuned checkpoints are saved under `training/checkpoints/`. The **Training** tab can start a run and show ROUGE, but the fine-tuned model is still not wired into the PDF summarization pipeline — it stays a side-by-side comparison.

## Running from the app

The backend exposes the experiment under `/api/training` and the frontend adds a **Training** tab:

- `GET /api/training/info` — device, default config, existing checkpoints, last ROUGE results.
- `POST /api/training/start` — launch a run in a background thread (one at a time).
- `GET /api/training/jobs/{id}` — poll status, current epoch, and per-epoch validation loss.
- `POST /api/training/jobs/{id}/cancel` — request a graceful stop.
- `POST /api/training/evaluate` — ROUGE for pretrained vs the latest fine-tuned checkpoint.

The shared logic lives in `training/service.py`, which both the CLI scripts and the API call. If the training extras are not installed, the tab shows a clear message instead of failing.

## Hardware notes

- **Batch size 1** with **gradient accumulation 16** (effective batch 16) to limit VRAM.
- **Max input length 512**, **max target length 200** tokens.
- **FP16** on CUDA when available; CPU training works but is slow.
- Default subset: **1000 train** + **100 validation** examples (arxiv config).
- Examples are collected via **streaming** (no full corpus download required).

## Setup

From the project root:

```bash
pip install -r training/requirements-training.txt
```

This installs `datasets` (2.x), `rouge-score`, and training dependencies **without** changing the main `requirements.txt`.

Note: `scientific_papers` requires `datasets>=2.14,<3` and `trust_remote_code=True` when loading (handled in `dataset_utils.py`).

## Dataset

- **Source:** `scientific_papers` on Hugging Face
- **Config:** `arxiv` (default; alternative: `pubmed`)
- **Input:** full article text, prefixed with `summarize: ` (T5 convention)
- **Target:** abstract (reference summary)
- Rows with empty article or abstract are removed

Both configs share the same article/abstract structure; `arxiv` covers physics, CS, and math papers, while `pubmed` covers biomedical literature. A larger PubMed run (5000 examples, 8 epochs) showed clearer ROUGE gains than a small arxiv baseline (1000 examples, 3 epochs).

## Train

```bash
python training/train_t5_small.py --train-size 1000 --val-size 100 --epochs 5
```

Quick dry-run (fewer examples, 2 epochs):

```bash
python training/train_t5_small.py --train-size 50 --val-size 10 --epochs 2 --logging-steps 10
```

Checkpoints are written to:

`training/checkpoints/t5-small-scientific/`

A final model is also saved to:

`training/checkpoints/t5-small-scientific/final/`

### Main training hyperparameters

| Parameter | Default |
|-----------|---------|
| Model | google-t5/t5-small |
| Dataset config | arxiv |
| Train examples | 1000 |
| Val examples | 100 |
| Batch size | 1 |
| Gradient accumulation | 16 |
| Learning rate | 5e-5 |
| Epochs | 5 |
| Max input tokens | 512 |
| Max target tokens | 200 |
| Best model selection | `load_best_model_at_end` on `eval_loss` (per-epoch save/eval) |

### Why 5 epochs

With 1000 training examples and an effective batch size of 16, each epoch is roughly 60 steps. Three epochs were not enough for the loss to settle. Five epochs give the validation loss room to flatten out so you can see whether fine-tuning helps, while staying short enough to avoid heavy overfitting on such a small subset. Because `load_best_model_at_end` is on, the saved model is the epoch with the lowest validation loss rather than just the last one. If you raise `--train-size`, more epochs may pay off; on 1000 examples, going past 7 mostly overfits.

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
├── service.py            # Shared training + ROUGE logic (CLI and API)
├── train_t5_small.py     # Fine-tuning CLI wrapper
├── evaluate_rouge.py     # ROUGE evaluation CLI wrapper
├── requirements-training.txt
├── README_training.md
├── checkpoints/          # Saved models (gitignored contents)
└── results/              # ROUGE JSON output
```

## Thesis usage

1. Describe this as **transfer learning / fine-tuning**, not training a new architecture from random weights.
2. Contrast with the app’s **DistilBART** (news-trained) and optional **Qwen** structured summaries.
3. Report ROUGE on the same validation slice before and after fine-tuning.
4. Discuss limitations: small subset, single domain (arxiv or pubmed), ROUGE-only evaluation.

## Limitations

- **Subset only** — not the full scientific_papers corpus.
- **ROUGE ≠ quality** — high overlap can still miss facts or produce dull text.
- **Comparison only** — the fine-tuned model is shown in the Training tab but not used for PDF summarization.
- **In-memory jobs** — a running job is lost if the backend restarts.
- **Download size** — first run downloads dataset and model weights from Hugging Face.

## Smoke test

```bash
python training/dataset_utils.py
```
