from unittest.mock import patch

import pytest

from haystack.nodes.prompt.invocation_layer import ChatGPTInvocationLayer


@pytest.mark.unit
def test_chatgpt_invoke_with_messages_as_prompt():
    with patch("haystack.nodes.prompt.invocation_layer.open_ai.load_openai_tokenizer"):
        layer = ChatGPTInvocationLayer(api_key="fake_key")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"},
    ]

    with patch("haystack.nodes.prompt.invocation_layer.chatgpt.openai_request") as mock_request:
        layer.invoke(prompt=messages)
        calls = mock_request.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["payload"]["messages"] == messages
