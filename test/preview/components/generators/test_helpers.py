from unittest.mock import Mock

import pytest

from haystack.preview.components.generators._helpers import enforce_token_limit, enforce_token_limit_chat


@pytest.mark.unit
def test_enforce_token_limit_above_limit(caplog):
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)

    assert enforce_token_limit("This is a test prompt.", tokenizer=tokenizer, max_tokens_limit=3) == "This is a"
    assert caplog.records[0].message == (
        "The prompt has been truncated from 5 tokens to 3 tokens to fit within the max token limit. "
        "Reduce the length of the prompt to prevent it from being cut off."
    )


@pytest.mark.unit
def test_enforce_token_limit_below_limit(caplog):
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)

    assert (
        enforce_token_limit("This is a test prompt.", tokenizer=tokenizer, max_tokens_limit=1000)
        == "This is a test prompt."
    )
    assert not caplog.records


@pytest.mark.unit
def test_enforce_token_limit_chat_above_limit(caplog):
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)

    assert enforce_token_limit_chat(
        ["System Prompt", "This is a test prompt."],
        tokenizer=tokenizer,
        max_tokens_limit=7,
        tokens_per_message_overhead=2,
    ) == ["System Prompt", "This is a"]
    assert caplog.records[0].message == (
        "The prompts have been truncated from 11 tokens to 7 tokens to fit within the max token limit. "
        "Reduce the length of the prompt to prevent it from being cut off."
    )


@pytest.mark.unit
def test_enforce_token_limit_chat_below_limit(caplog):
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)

    assert enforce_token_limit_chat(
        ["System Prompt", "This is a test prompt."],
        tokenizer=tokenizer,
        max_tokens_limit=100,
        tokens_per_message_overhead=2,
    ) == ["System Prompt", "This is a test prompt."]
    assert not caplog.records
