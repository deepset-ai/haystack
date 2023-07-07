from unittest.mock import patch

import logging
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


@pytest.mark.unit
def test_chatgpt_token_limit_warning_single_prompt(mock_openai_tokenizer, caplog):
    invocation_layer = ChatGPTInvocationLayer(
        model_name_or_path="gpt-3.5-turbo",
        api_key="fake_api_key",
        api_base="https://fake_api_base.com",
        max_length=4090,
    )
    with caplog.at_level(logging.WARNING):
        _ = invocation_layer._ensure_token_limit(prompt="This is a test for a mock openai tokenizer.")
        assert "The prompt has been truncated from" in caplog.text
        assert "and answer length (4090 tokens) fit within the max token limit (4096 tokens)." in caplog.text


@pytest.mark.unit
def test_chatgpt_token_limit_warning_with_messages(mock_openai_tokenizer, caplog):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"},
    ]
    with patch("haystack.utils.openai_utils.count_openai_tokens_messages") as mock_count_tokens:
        mock_count_tokens.return_value = 40
        invocation_layer = ChatGPTInvocationLayer(
            model_name_or_path="gpt-3.5-turbo",
            api_key="fake_api_key",
            api_base="https://fake_api_base.com",
            max_length=4060,
        )
        with pytest.raises(ValueError):
            _ = invocation_layer._ensure_token_limit(prompt=messages)
