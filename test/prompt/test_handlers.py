from unittest.mock import patch

import pytest

from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler


@pytest.mark.unit
def test_prompt_handler_positive():
    # prompt of length 5 + max_length of 3 = 8, which is less than model_max_length of 10, so no resize
    mock_tokens = ["I", "am", "a", "tokenized", "prompt"]
    mock_prompt = "I am a tokenized prompt"

    with patch(
        "haystack.nodes.prompt.invocation_layer.handlers.AutoTokenizer.from_pretrained", autospec=True
    ) as mock_tokenizer:
        tokenizer_instance = mock_tokenizer.return_value
        tokenizer_instance.tokenize.return_value = mock_tokens
        tokenizer_instance.convert_tokens_to_string.return_value = mock_prompt

        prompt_handler = DefaultPromptHandler("model_path", 10, 3)

        # Test with a prompt that does not exceed model_max_length when tokenized
        result = prompt_handler(mock_prompt)

    assert result == {
        "resized_prompt": mock_prompt,
        "prompt_length": 5,
        "new_prompt_length": 5,
        "model_max_length": 10,
        "max_length": 3,
    }


@pytest.mark.unit
def test_prompt_handler_negative():
    # prompt of length 8 + max_length of 3 = 11, which is more than model_max_length of 10, so we resize to 7
    mock_tokens = ["I", "am", "a", "tokenized", "prompt", "of", "length", "eight"]
    mock_prompt = "I am a tokenized prompt of length"

    with patch(
        "haystack.nodes.prompt.invocation_layer.handlers.AutoTokenizer.from_pretrained", autospec=True
    ) as mock_tokenizer:
        tokenizer_instance = mock_tokenizer.return_value
        tokenizer_instance.tokenize.return_value = mock_tokens
        tokenizer_instance.convert_tokens_to_string.return_value = mock_prompt

        prompt_handler = DefaultPromptHandler("model_path", 10, 3)
        result = prompt_handler(mock_prompt)

    assert result == {
        "resized_prompt": mock_prompt,
        "prompt_length": 8,
        "new_prompt_length": 7,
        "model_max_length": 10,
        "max_length": 3,
    }


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
