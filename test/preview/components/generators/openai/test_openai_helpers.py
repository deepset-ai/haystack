import pytest

from haystack.preview.components.generators.openai._helpers import enforce_token_limit


@pytest.mark.unit
def test_enforce_token_limit_above_limit(caplog, mock_tokenizer):
    prompt = enforce_token_limit("This is a test prompt.", tokenizer=mock_tokenizer, max_tokens_limit=3)
    assert prompt == "This is a"
    assert caplog.records[0].message == (
        "The prompt has been truncated from 5 tokens to 3 tokens to fit within the max token "
        "limit. Reduce the length of the prompt to prevent it from being cut off."
    )


@pytest.mark.unit
def test_enforce_token_limit_below_limit(caplog, mock_tokenizer):
    prompt = enforce_token_limit("This is a test prompt.", tokenizer=mock_tokenizer, max_tokens_limit=100)
    assert prompt == "This is a test prompt."
    assert not caplog.records
