from unittest.mock import patch

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
