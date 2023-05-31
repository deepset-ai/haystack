from unittest.mock import patch, Mock, call
import json
import os

import pytest

from haystack.nodes.prompt.invocation_layer.handlers import DefaultTokenStreamingHandler
from haystack.nodes.prompt.invocation_layer import AnthropicClaudeInvocationLayer


@pytest.mark.unit
def test_default_costuctor():
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.Tokenizer"):
        layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key")

    assert layer.api_key == "some_fake_key"
    assert layer.max_length == 200
    assert layer.max_tokens_limit == 9000
    assert layer.model_input_kwargs == {}


@pytest.mark.unit
def test_ignored_kwargs_are_filtered_in_init():
    kwargs = {
        "temperature": 1,
        "top_p": 5,
        "top_k": 2,
        "stop_sequences": ["\n\nHuman: "],
        "stream": True,
        "stream_handler": DefaultTokenStreamingHandler(),
        "unkwnown_args": "this will be filtered out",
    }
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.Tokenizer"):
        layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key", **kwargs)

    # Verify unexpected kwargs are filtered out
    assert len(layer.model_input_kwargs) == 6
    assert "temperature" in layer.model_input_kwargs
    assert "top_p" in layer.model_input_kwargs
    assert "top_k" in layer.model_input_kwargs
    assert "stop_sequences" in layer.model_input_kwargs
    assert "stream" in layer.model_input_kwargs
    assert "stream_handler" in layer.model_input_kwargs
    assert "unkwnown_args" not in layer.model_input_kwargs


@pytest.mark.unit
def test_invoke_with_no_kwargs():
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.Tokenizer"):
        layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key")

    with pytest.raises(ValueError) as e:
        layer.invoke()
        assert e.match("No prompt provided.")


@pytest.mark.unit
@patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.request_with_retry")
def test_invoke_with_prompt_only(mock_request):
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.Tokenizer"):
        layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key")

    # Create a fake response
    mock_response = Mock(**{"status_code": 200, "ok": True, "json.return_value": {"completion": "some_result "}})
    mock_request.return_value = mock_response

    res = layer.invoke(prompt="Some prompt")
    assert len(res) == 1
    assert res[0] == "some_result"


@pytest.mark.unit
def test_invoke_with_kwargs():
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.Tokenizer"):
        layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key")

    # Create a fake response
    mock_response = Mock(**{"status_code": 200, "ok": True, "json.return_value": {"completion": "some_result "}})
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.request_with_retry") as mock_request:
        mock_request.return_value = mock_response
        res = layer.invoke(prompt="Some prompt", max_length=300, stop_words=["stop", "here"])
    assert len(res) == 1
    assert res[0] == "some_result"

    expected_data = {
        "model": "claude-v1",
        "prompt": "\n\nHuman: Some prompt\n\nAssistant: ",
        "max_tokens_to_sample": 300,
        "temperature": 1,
        "top_p": -1,
        "top_k": -1,
        "stream": False,
        "stop_sequences": ["stop", "here", "\n\nHuman: "],
    }
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["data"] == json.dumps(expected_data)


@pytest.mark.unit
def test_invoke_with_none_stop_words():
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.Tokenizer"):
        layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key")

    # Create a fake response
    mock_response = Mock(**{"status_code": 200, "ok": True, "json.return_value": {"completion": "some_result "}})
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.request_with_retry") as mock_request:
        mock_request.return_value = mock_response
        res = layer.invoke(prompt="Some prompt", max_length=300, stop_words=None)
    assert len(res) == 1
    assert res[0] == "some_result"

    expected_data = {
        "model": "claude-v1",
        "prompt": "\n\nHuman: Some prompt\n\nAssistant: ",
        "max_tokens_to_sample": 300,
        "temperature": 1,
        "top_p": -1,
        "top_k": -1,
        "stream": False,
        "stop_sequences": ["\n\nHuman: "],
    }
    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["data"] == json.dumps(expected_data)


@pytest.mark.unit
def test_invoke_with_stream():
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.Tokenizer"):
        layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key")

    # Create a fake streamed response
    def mock_iter(self):
        fake_data = json.dumps({"completion": " The sky appears"})
        yield f"data: {fake_data}\n\n".encode()
        fake_data = json.dumps({"completion": " The sky appears blue to"})
        yield f"data: {fake_data}\n\n".encode()
        fake_data = json.dumps({"completion": " The sky appears blue to us due to how"})
        yield f"data: {fake_data}\n\n".encode()
        yield "data: [DONE]\n\n".encode()

    mock_response = Mock(**{"__iter__": mock_iter})

    # Verifies expected result is returned
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.request_with_retry") as mock_request:
        mock_request.return_value = mock_response
        res = layer.invoke(prompt="Some prompt", stream=True)

    assert len(res) == 1
    assert res[0] == " The sky appears blue to us due to how"


@pytest.mark.unit
def test_invoke_with_custom_stream_handler():
    # Create a mock stream handler that always return the same token when called
    mock_stream_handler = Mock()
    mock_stream_handler.return_value = "token"

    # Create a layer with a mocked stream handler
    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.Tokenizer"):
        layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key", stream_handler=mock_stream_handler)

    # Create a fake streamed response
    def mock_iter(self):
        fake_data = json.dumps({"completion": " The sky appears"})
        yield f"data: {fake_data}\n\n".encode()
        fake_data = json.dumps({"completion": " The sky appears blue to"})
        yield f"data: {fake_data}\n\n".encode()
        fake_data = json.dumps({"completion": " The sky appears blue to us due to how"})
        yield f"data: {fake_data}\n\n".encode()
        yield "data: [DONE]\n\n".encode()

    mock_response = Mock(**{"__iter__": mock_iter})

    with patch("haystack.nodes.prompt.invocation_layer.anthropic_claude.request_with_retry") as mock_request:
        mock_request.return_value = mock_response
        res = layer.invoke(prompt="Some prompt")

    assert len(res) == 1
    # This is not the real result but the values returned by the mock handler
    assert res[0] == " The sky appears blue to us due to how"

    # Verifies the handler has been called the expected times with the expected args
    assert mock_stream_handler.call_count == 3
    expected_call_list = [call(" The sky appears"), call(" blue to"), call(" us due to how")]
    assert mock_stream_handler.call_args_list == expected_call_list


@pytest.mark.unit
def test_ensure_token_limit_fails_if_called_with_list():
    layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key")
    with pytest.raises(ValueError):
        layer._ensure_token_limit(prompt=[])


@pytest.mark.integration
def test_ensure_token_limit_with_small_max_length(caplog):
    layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key", max_length=10)
    res = layer._ensure_token_limit(prompt="Short prompt")

    assert res == "Short prompt"
    assert not caplog.records

    res = layer._ensure_token_limit(prompt="This is a very very very very very much longer prompt")
    assert res == "This is a very very very very very much longer prompt"
    assert not caplog.records


@pytest.mark.integration
def test_ensure_token_limit_with_huge_max_length(caplog):
    layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key", max_length=8990)
    res = layer._ensure_token_limit(prompt="Short prompt")

    assert res == "Short prompt"
    assert not caplog.records

    res = layer._ensure_token_limit(prompt="This is a very very very very very much longer prompt")
    assert res == "This is a very very very very very much longer"
    assert len(caplog.records) == 1
    expected_message_log = (
        "The prompt has been truncated from 11 tokens to 10 tokens so that the prompt length and "
        "answer length (8990 tokens) fits within the max token limit (9000 tokens). "
        "Reduce the length of the prompt to prevent it from being cut off."
    )
    assert caplog.records[0].message == expected_message_log


@pytest.mark.unit
def test_supports():
    layer = AnthropicClaudeInvocationLayer(api_key="some_fake_key")

    assert not layer.supports("claude")

    assert layer.supports("claude-v1")
    assert layer.supports("claude-v1.0")
    assert layer.supports("claude-v1.2")
    assert layer.supports("claude-v1.3")
    assert not layer.supports("claude-v2.0")
    assert layer.supports("claude-instant-v1")
    assert layer.supports("claude-instant-v1.0")
    assert layer.supports("claude-instant-v1.1")
    assert not layer.supports("claude-instant-v2.0")


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ANTHROPIC_CLAUDE_API_KEY", "") == "", reason="Anthropic Claude API key not found")
def test_invoke_non_streamed():
    api_key = os.environ.get("ANTHROPIC_CLAUDE_API_KEY")
    layer = AnthropicClaudeInvocationLayer(api_key=api_key)

    res = layer.invoke(prompt="Why is the sky blue?")

    # Verifies answer has been received
    assert len(res) == 1


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ANTHROPIC_CLAUDE_API_KEY", "") == "", reason="Anthropic Claude API key not found")
def test_invoke_streamed():
    api_key = os.environ.get("ANTHROPIC_CLAUDE_API_KEY")
    layer = AnthropicClaudeInvocationLayer(api_key=api_key)

    res = layer.invoke(prompt="Why is the sky blue?", stream=True)

    # Verifies answer has been received
    assert len(res) == 1
