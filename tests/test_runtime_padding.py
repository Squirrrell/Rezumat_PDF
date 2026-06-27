"""Tests for runtime padding helpers."""

from __future__ import annotations

from unittest.mock import patch

from backend.runtime_padding import pad_runtime_to_interval


@patch("backend.runtime_padding.time.sleep")
@patch("backend.runtime_padding.time.perf_counter", return_value=170.0)
def test_pad_runtime_skips_sleep_when_above_minimum(mock_perf_counter, mock_sleep):
    result = pad_runtime_to_interval(100.0, 60.0, 80.0)
    assert result == 70.0
    mock_sleep.assert_not_called()


@patch("backend.runtime_padding.time.sleep")
@patch("backend.runtime_padding.time.perf_counter")
def test_pad_runtime_sleeps_when_below_minimum(mock_perf_counter, mock_sleep):
    mock_perf_counter.side_effect = [105.0, 165.0]
    with patch("backend.runtime_padding.random.uniform", return_value=65.0):
        result = pad_runtime_to_interval(100.0, 60.0, 80.0)
    mock_sleep.assert_called_once_with(60.0)
    assert result == 65.0
