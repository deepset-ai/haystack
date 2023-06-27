from unittest.mock import patch

import logging
import pytest

from haystack.nodes.prompt.invocation_layer import OpenAIInvocationLayer


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.open_ai.openai_request")
def test_default_api_base(mock_request):
    with patch("haystack.nodes.prompt.invocation_layer.open_ai.load_openai_tokenizer"):
        invocation_layer = OpenAIInvocationLayer(api_key="fake_api_key")
    assert invocation_layer.api_base == "https://api.openai.com/v1"
    assert invocation_layer.url == "https://api.openai.com/v1/completions"

    invocation_layer.invoke(prompt="dummy_prompt")
    assert mock_request.call_args.kwargs["url"] == "https://api.openai.com/v1/completions"


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.open_ai.openai_request")
def test_custom_api_base(mock_request):
    with patch("haystack.nodes.prompt.invocation_layer.open_ai.load_openai_tokenizer"):
        invocation_layer = OpenAIInvocationLayer(api_key="fake_api_key", api_base="https://fake_api_base.com")
    assert invocation_layer.api_base == "https://fake_api_base.com"
    assert invocation_layer.url == "https://fake_api_base.com/completions"

    invocation_layer.invoke(prompt="dummy_prompt")
    assert mock_request.call_args.kwargs["url"] == "https://fake_api_base.com/completions"


@pytest.mark.unit
def test_openai_token_limit_warning(mock_openai_tokenizer, caplog):
    invocation_layer = OpenAIInvocationLayer(
        model_name_or_path="text-ada-001", api_key="fake_api_key", api_base="https://fake_api_base.com", max_length=2045
    )
    with caplog.at_level(logging.WARNING):
        _ = invocation_layer._ensure_token_limit(prompt="This is a test for a mock openai tokenizer.")
        assert "The prompt has been truncated from" in caplog.text
        assert "and answer length (2045 tokens) fit within the max token limit (2049 tokens)." in caplog.text
