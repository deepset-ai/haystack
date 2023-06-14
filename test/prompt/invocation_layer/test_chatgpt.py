from unittest.mock import patch

import pytest

from haystack.nodes.prompt.invocation_layer import ChatGPTInvocationLayer


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.chatgpt.openai_request")
def test_default_api_base(mock_request):
    with patch("haystack.nodes.prompt.invocation_layer.open_ai.load_openai_tokenizer"):
        invocation_layer = ChatGPTInvocationLayer(api_key="fake_api_key")
    assert invocation_layer.api_base == "https://api.openai.com/v1"
    assert invocation_layer.url == "https://api.openai.com/v1/chat/completions"

    invocation_layer.invoke(prompt="dummy_prompt")
    assert mock_request.call_args.kwargs["url"] == "https://api.openai.com/v1/chat/completions"


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.chatgpt.openai_request")
def test_custom_api_base(mock_request):
    with patch("haystack.nodes.prompt.invocation_layer.open_ai.load_openai_tokenizer"):
        invocation_layer = ChatGPTInvocationLayer(api_key="fake_api_key", api_base="https://fake_api_base.com")
    assert invocation_layer.api_base == "https://fake_api_base.com"
    assert invocation_layer.url == "https://fake_api_base.com/chat/completions"

    invocation_layer.invoke(prompt="dummy_prompt")
    assert mock_request.call_args.kwargs["url"] == "https://fake_api_base.com/chat/completions"


@pytest.mark.unit
def test_supports_correct_model_names():
    for model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613"]:
        assert ChatGPTInvocationLayer.supports(model_name)


@pytest.mark.unit
def test_does_not_support_wrong_model_names():
    for model_name in ["got-3.5-turbo", "wrong_model_name"]:
        assert not ChatGPTInvocationLayer.supports(model_name)
