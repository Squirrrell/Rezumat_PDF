"""Dataset loading and preprocessing for T5 fine-tuning on scientific papers."""

from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
)

DEFAULT_MODEL_NAME = "google-t5/t5-small"
DEFAULT_DATASET_CONFIG = "arxiv"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
T5_PREFIX = "summarize: "


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> AutoTokenizer:
    """Load T5 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def preprocess_function(
    examples: dict,
    tokenizer: AutoTokenizer,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
) -> dict:
    """Tokenize article (input) and abstract (target) for seq2seq training."""
    inputs = [T5_PREFIX + (a or "") for a in examples["article"]]
    targets = examples["abstract"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_data_collator(
    tokenizer: AutoTokenizer,
    model: PreTrainedModel,
) -> DataCollatorForSeq2Seq:
    """Collator with dynamic padding for seq2seq."""
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )


def _filter_nonempty(example: dict) -> bool:
    article = (example.get("article") or "").strip()
    abstract = (example.get("abstract") or "").strip()
    return bool(article) and bool(abstract)


def _collect_subset_streaming(
    config_name: str,
    total_needed: int,
    max_scan: int | None = None,
) -> Dataset:
    """
    Stream the dataset and collect valid rows without downloading the full corpus.

    max_scan: stop scanning after this many raw rows (default 15 * total_needed).
    """
    if max_scan is None:
        max_scan = max(total_needed * 15, total_needed + 50)

    stream = load_dataset(
        "scientific_papers",
        config_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    collected: list[dict] = []
    scanned = 0
    for row in stream:
        scanned += 1
        if _filter_nonempty(row):
            collected.append(
                {"article": row["article"], "abstract": row["abstract"]}
            )
        if len(collected) >= total_needed:
            break
        if scanned >= max_scan:
            break

    if len(collected) < total_needed:
        raise ValueError(
            f"Only found {len(collected)} valid examples after scanning {scanned} rows; "
            f"need {total_needed}. Increase max_scan or check the dataset."
        )

    return Dataset.from_list(collected)


def load_scientific_papers_subset(
    config_name: str = DEFAULT_DATASET_CONFIG,
    train_size: int = 1000,
    val_size: int = 100,
    model_name: str = DEFAULT_MODEL_NAME,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
) -> tuple[Dataset, Dataset, AutoTokenizer]:
    """
    Load train/validation subsets from Hugging Face scientific_papers.

    Uses article as input and abstract as target summary.
    Streams data so the full 3.6GB arxiv split is not required locally.
    """
    tokenizer = load_tokenizer(model_name)

    total_needed = train_size + val_size
    subset = _collect_subset_streaming(config_name, total_needed)
    train_raw = subset.select(range(train_size))
    val_raw = subset.select(range(train_size, train_size + val_size))

    def _tokenize(batch):
        return preprocess_function(
            batch,
            tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )

    train_dataset = train_raw.map(
        _tokenize,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train",
    )
    val_dataset = val_raw.map(
        _tokenize,
        batched=True,
        remove_columns=val_raw.column_names,
        desc="Tokenizing validation",
    )

    return train_dataset, val_dataset, tokenizer


def load_eval_raw_subset(
    config_name: str = DEFAULT_DATASET_CONFIG,
    train_size: int = 1000,
    val_size: int = 100,
) -> Dataset:
    """Load raw (non-tokenized) validation rows for ROUGE evaluation."""
    total_needed = train_size + val_size
    subset = _collect_subset_streaming(config_name, total_needed)
    return subset.select(range(train_size, train_size + val_size))


if __name__ == "__main__":
    print("Loading 10 train + 5 val examples (smoke test)...")
    train_ds, val_ds, tok = load_scientific_papers_subset(
        train_size=10,
        val_size=5,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Sample input ids length: {len(train_ds[0]['input_ids'])}")
    print("dataset_utils smoke test OK")
