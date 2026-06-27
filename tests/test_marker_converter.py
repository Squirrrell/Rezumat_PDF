"""Unit tests for Marker converter profile and config."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.marker_converter import (
    FAST_PROCESSOR_PATHS,
    _create_artifact_dict,
    _marker_config,
    marker_profile,
)


@pytest.fixture(autouse=True)
def clear_marker_env(monkeypatch):
    for key in (
        "MARKER_PROFILE",
        "MARKER_LAYOUT_BATCH_SIZE",
        "MARKER_DETECTION_BATCH_SIZE",
        "MARKER_OCR_ERROR_BATCH_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_marker_profile_defaults_to_fast():
    assert marker_profile() == "fast"


def test_marker_profile_quality_env():
    os.environ["MARKER_PROFILE"] = "quality"
    assert marker_profile() == "quality"


@patch("src.marker_converter.cuda_device_info", return_value={"gpu_memory_gb": 4.0})
def test_fast_config_uses_low_vram_batches(mock_cuda):
    config = _marker_config("fast", "cuda")
    assert config["layout_batch_size"] == 8
    assert config["detection_batch_size"] == 4
    assert config["ocr_error_batch_size"] == 6
    assert config["lowres_image_dpi"] == 72
    assert config["disable_ocr"] is True
    assert config["extract_images"] is False


@patch("src.marker_converter.cuda_device_info", return_value={"gpu_memory_gb": 12.0})
def test_fast_config_uses_high_vram_batches(mock_cuda):
    config = _marker_config("fast", "cuda")
    assert config["layout_batch_size"] == 12
    assert config["detection_batch_size"] == 8


def test_quality_config_skips_default_batch_sizes_on_cuda():
    config = _marker_config("quality", "cuda")
    assert "layout_batch_size" not in config
    assert config["disable_tqdm"] is True


@patch("src.marker_converter.cuda_device_info", return_value={"gpu_memory_gb": 4.0})
def test_env_overrides_batch_sizes(mock_cuda):
    os.environ["MARKER_LAYOUT_BATCH_SIZE"] = "4"
    config = _marker_config("fast", "cuda")
    assert config["layout_batch_size"] == 4


def test_fast_artifact_dict_has_four_models():
    keys = _create_artifact_dict("cpu", None, "fast").keys()
    assert set(keys) == {
        "layout_model",
        "recognition_model",
        "detection_model",
        "ocr_error_model",
    }


def test_quality_artifact_dict_has_five_models():
    with patch("src.marker_converter.create_model_dict") as mock_create:
        mock_create.return_value = {
            "layout_model": object(),
            "recognition_model": object(),
            "table_rec_model": object(),
            "detection_model": object(),
            "ocr_error_model": object(),
        }
        keys = _create_artifact_dict("cuda", None, "quality").keys()
        assert len(keys) == 5
        mock_create.assert_called_once()


def test_fast_processor_list_length():
    assert len(FAST_PROCESSOR_PATHS) == 13
    assert all(path.startswith("marker.processors.") for path in FAST_PROCESSOR_PATHS)
