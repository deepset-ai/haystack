from unittest.mock import patch

import pytest

from haystack.nodes.prompt.invocation_layer.handlers import (
    DefaultTokenStreamingHandler,
    DefaultPromptHandler,
    AnthropicTokenStreamingHandler,
)


@pytest.mark.integration
def test_prompt_handler_basics():
    handler = DefaultPromptHandler(model_name_or_path="gpt2", model_max_length=20, max_length=10)
    assert callable(handler)

    handler = DefaultPromptHandler(model_name_or_path="gpt2", model_max_length=20)
    assert handler.max_length == 100


@pytest.mark.integration
def test_gpt2_prompt_handler():
    # test gpt2 BPE based tokenizer
    handler = DefaultPromptHandler(model_name_or_path="gpt2", model_max_length=20, max_length=10)

    # test no resize
    assert handler("This is a test") == {
        "prompt_length": 4,
        "resized_prompt": "This is a test",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 4,
    }

    # test resize
    assert handler("This is a prompt that will be resized because it is longer than allowed") == {
        "prompt_length": 15,
        "resized_prompt": "This is a prompt that will be resized because",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 10,
    }


@pytest.mark.integration
def test_flan_prompt_handler_no_resize():
    handler = DefaultPromptHandler(model_name_or_path="google/flan-t5-xxl", model_max_length=20, max_length=10)
    assert handler("This is a test") == {
        "prompt_length": 5,
        "resized_prompt": "This is a test",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 5,
    }


@pytest.mark.integration
def test_flan_prompt_handler_resize():
    handler = DefaultPromptHandler(model_name_or_path="google/flan-t5-xxl", model_max_length=20, max_length=10)
    assert handler("This is a prompt that will be resized because it is longer than allowed") == {
        "prompt_length": 17,
        "resized_prompt": "This is a prompt that will be re",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 10,
    }


@pytest.mark.integration
def test_flan_prompt_handler_empty_string():
    handler = DefaultPromptHandler(model_name_or_path="google/flan-t5-xxl", model_max_length=20, max_length=10)
    assert handler("") == {
        "prompt_length": 0,
        "resized_prompt": "",
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 0,
    }


@pytest.mark.integration
def test_flan_prompt_handler_none():
    handler = DefaultPromptHandler(model_name_or_path="google/flan-t5-xxl", model_max_length=20, max_length=10)
    assert handler(None) == {
        "prompt_length": 0,
        "resized_prompt": None,
        "max_length": 10,
        "model_max_length": 20,
        "new_prompt_length": 0,
    }


@pytest.mark.unit
@patch("builtins.print")
def test_anthropic_token_streaming_handler(mock_print):
    handler = AnthropicTokenStreamingHandler(DefaultTokenStreamingHandler())

    res = handler(" This")
    assert res == " This"
    mock_print.assert_called_with(" This", flush=True, end="")

    res = handler(" This is a new")
    assert res == " is a new"
    mock_print.assert_called_with(" is a new", flush=True, end="")

    res = handler(" This is a new token")
    assert res == " token"
    mock_print.assert_called_with(" token", flush=True, end="")

    res = handler("And now")
    assert res == "And now"
    mock_print.assert_called_with("And now", flush=True, end="")

    res = handler("And now something completely different")
    assert res == " something completely different"
    mock_print.assert_called_with(" something completely different", flush=True, end="")
