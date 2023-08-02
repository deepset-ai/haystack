from unittest.mock import patch

import logging
import pytest

from haystack.nodes.prompt.invocation_layer import OpenAIInvocationLayer


@pytest.fixture
def load_openai_tokenizer():
    with patch("haystack.nodes.prompt.invocation_layer.open_ai.load_openai_tokenizer") as mock_load_openai_tokenizer:
        yield mock_load_openai_tokenizer


@pytest.fixture()
def mock_open_ai_request():
    with patch("haystack.nodes.prompt.invocation_layer.open_ai.openai_request") as mock_openai_request:
        yield mock_openai_request


@pytest.mark.unit
def test_default_api_base(mock_open_ai_request, load_openai_tokenizer):
    invocation_layer = OpenAIInvocationLayer(api_key="fake_api_key")
    assert invocation_layer.api_base == "https://api.openai.com/v1"
    assert invocation_layer.url == "https://api.openai.com/v1/completions"

    invocation_layer.invoke(prompt="dummy_prompt")
    assert mock_open_ai_request.call_args.kwargs["url"] == "https://api.openai.com/v1/completions"


@pytest.mark.unit
def test_custom_api_base(mock_open_ai_request, load_openai_tokenizer):
    invocation_layer = OpenAIInvocationLayer(api_key="fake_api_key", api_base="https://fake_api_base.com")
    assert invocation_layer.api_base == "https://fake_api_base.com"
    assert invocation_layer.url == "https://fake_api_base.com/completions"

    invocation_layer.invoke(prompt="dummy_prompt")
    assert mock_open_ai_request.call_args.kwargs["url"] == "https://fake_api_base.com/completions"


@pytest.mark.unit
def test_openai_token_limit_warning(mock_openai_tokenizer, caplog):
    invocation_layer = OpenAIInvocationLayer(
        model_name_or_path="text-ada-001", api_key="fake_api_key", api_base="https://fake_api_base.com", max_length=2045
    )
    with caplog.at_level(logging.WARNING):
        _ = invocation_layer._ensure_token_limit(prompt="This is a test for a mock openai tokenizer.")
        assert "The prompt has been truncated from" in caplog.text
        assert "and answer length (2045 tokens) fit within the max token limit (2049 tokens)." in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name,max_tokens_limit",
    [
        ("text-davinci-003", 4097),
        ("gpt-3.5-turbo", 4096),
        ("gpt-3.5-turbo-16k", 16384),
        ("gpt-4-32k", 32768),
        ("gpt-4", 8192),
    ],
)
def test_openai_token_limit_warning_not_triggered(caplog, mock_openai_tokenizer, model_name, max_tokens_limit):
    layer = OpenAIInvocationLayer(
        model_name_or_path=model_name, api_key="fake_api_key", api_base="https://fake_api_base.com", max_length=256
    )

    assert layer.max_tokens_limit == max_tokens_limit

    # the warning is not triggered because max_length is 256, our prompt is 11 tokens, and we have big context window
    _ = layer._ensure_token_limit(prompt="This is a test for a mock openai tokenizer.")
    assert not caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name,max_tokens_limit",
    [
        ("text-davinci-003", 4097),
        ("gpt-3.5-turbo", 4096),
        ("gpt-3.5-turbo-16k", 16384),
        ("gpt-4-32k", 32768),
        ("gpt-4", 8192),
    ],
)
def test_openai_token_limit_warning_is_triggered(caplog, mock_openai_tokenizer, model_name, max_tokens_limit):
    layer = OpenAIInvocationLayer(
        model_name_or_path=model_name,
        api_key="fake_api_key",
        api_base="https://fake_api_base.com",
        max_length=int(max_tokens_limit) - 1,
    )

    assert layer.max_tokens_limit == max_tokens_limit

    # the warning is triggered because max_length is one token smaller than context window and our prompt has 11 tokens
    _ = layer._ensure_token_limit(prompt="This is a test for a mock openai tokenizer.")

    # since we are truncating the prompt of 11 tokens, we should see a warning that only 1 token is left
    assert "The prompt has been truncated from 11 tokens to 1 tokens" in caplog.text


@pytest.mark.unit
def test_no_openai_organization(mock_open_ai_request, load_openai_tokenizer):
    invocation_layer = OpenAIInvocationLayer(api_key="fake_api_key")

    assert invocation_layer.openai_organization is None
    assert "OpenAI-Organization" not in invocation_layer.headers

    invocation_layer.invoke(prompt="dummy_prompt")
    assert "OpenAI-Organization" not in mock_open_ai_request.call_args.kwargs["headers"]


@pytest.mark.unit
def test_openai_organization(mock_open_ai_request, load_openai_tokenizer):
    invocation_layer = OpenAIInvocationLayer(api_key="fake_api_key", openai_organization="fake_organization")

    assert invocation_layer.openai_organization == "fake_organization"
    assert invocation_layer.headers["OpenAI-Organization"] == "fake_organization"

    invocation_layer.invoke(prompt="dummy_prompt")
    assert mock_open_ai_request.call_args.kwargs["headers"]["OpenAI-Organization"] == "fake_organization"


@pytest.mark.unit
def test_supports(load_openai_tokenizer):
    layer = OpenAIInvocationLayer(api_key="some_fake_key")

    assert layer.supports("ada")
    assert layer.supports("babbage")
    assert layer.supports("curie")
    assert layer.supports("davinci")
    assert layer.supports("text-ada-001")
    assert layer.supports("text-davinci-002")

    # the following model contains "ada" in the name, but it's not from OpenAI
    assert not layer.supports("ybelkada/mpt-7b-bf16-sharded")
