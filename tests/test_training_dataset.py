"""Unit tests for the training dataset utilities (no network or heavy deps)."""

from __future__ import annotations

from unittest.mock import MagicMock

from training.dataset_utils import (
    DEFAULT_DATASET_CONFIG,
    T5_PREFIX,
    _filter_nonempty,
    preprocess_function,
)


def test_default_dataset_config_is_arxiv():
    assert DEFAULT_DATASET_CONFIG == "arxiv"


def test_filter_nonempty():
    assert _filter_nonempty({"article": "a", "abstract": "b"})
    assert not _filter_nonempty({"article": "  ", "abstract": "b"})
    assert not _filter_nonempty({"article": "a", "abstract": ""})
    assert not _filter_nonempty({})


def test_preprocess_adds_t5_prefix():
    tokenizer = MagicMock()
    examples = {"article": ["hello", None], "abstract": ["sum a", "sum b"]}

    preprocess_function(examples, tokenizer)

    input_arg = tokenizer.call_args_list[0].args[0]
    assert input_arg == [f"{T5_PREFIX}hello", T5_PREFIX]

    target_kwargs = tokenizer.call_args_list[1].kwargs
    assert target_kwargs["text_target"] == ["sum a", "sum b"]
