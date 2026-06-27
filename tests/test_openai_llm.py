"""Unit tests for OpenAI API parameter selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.llm_config import openai_supports_temperature, openai_uses_completion_tokens
from src.openai_llm import _build_chat_kwargs, chat_completion


def test_openai_uses_completion_tokens_for_gpt5_and_o_series():
    assert openai_uses_completion_tokens("gpt-5-mini") is True
    assert openai_uses_completion_tokens("gpt-5-nano-2025-08-07") is True
    assert openai_uses_completion_tokens("o1-mini") is True
    assert openai_uses_completion_tokens("o3-mini") is True
    assert openai_uses_completion_tokens("o4-mini") is True
    assert openai_uses_completion_tokens("gpt-4o-mini") is False
    assert openai_uses_completion_tokens("gpt-4o") is False


def test_openai_supports_temperature_legacy_only():
    assert openai_supports_temperature("gpt-4o-mini") is True
    assert openai_supports_temperature("gpt-5-mini") is False


def test_build_chat_kwargs_gpt5_mini():
    kwargs = _build_chat_kwargs("gpt-5-mini", "hello", max_tokens=250)
    assert kwargs["model"] == "gpt-5-mini"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["max_completion_tokens"] == 250
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


def test_build_chat_kwargs_gpt4o_mini():
    kwargs = _build_chat_kwargs("gpt-4o-mini", "hello", max_tokens=400)
    assert kwargs["max_tokens"] == 400
    assert kwargs["temperature"] == 0.2
    assert "max_completion_tokens" not in kwargs


@patch("src.openai_llm._get_client")
@patch("src.openai_llm.get_openai_model", return_value="gpt-5-mini")
def test_chat_completion_uses_completion_tokens_for_gpt5(mock_model, mock_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="answer"))]
    create = MagicMock(return_value=mock_response)
    mock_client.return_value.chat.completions.create = create

    result = chat_completion("test prompt", max_tokens=320)

    assert result == "answer"
    kwargs = create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 320
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


@patch("src.openai_llm._get_client")
@patch("src.openai_llm.get_openai_model", return_value="gpt-4o-mini")
def test_chat_completion_uses_max_tokens_for_legacy(mock_model, mock_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="legacy"))]
    create = MagicMock(return_value=mock_response)
    mock_client.return_value.chat.completions.create = create

    result = chat_completion("test prompt", max_tokens=400)

    assert result == "legacy"
    kwargs = create.call_args.kwargs
    assert kwargs["max_tokens"] == 400
    assert kwargs["temperature"] == 0.2
    assert "max_completion_tokens" not in kwargs
