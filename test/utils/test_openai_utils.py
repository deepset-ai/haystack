import pytest

from haystack.utils.openai_utils import _openai_text_completion_tokenization_details


@pytest.mark.unit
def test_openai_text_completion_tokenization_details():
    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="gpt-35-turbo")
    assert tokenizer_name == "cl100k_base"
    assert max_tokens_limit == 4096

    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="gpt-3.5-turbo")
    assert tokenizer_name == "cl100k_base"
    assert max_tokens_limit == 4096

    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="text-davinci-003")
    assert tokenizer_name == "p50k_base"
    assert max_tokens_limit == 4097

    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="gpt-4")
    assert tokenizer_name == "cl100k_base"
    assert max_tokens_limit == 8192

    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="gpt-4-32k")
    assert tokenizer_name == "cl100k_base"
    assert max_tokens_limit == 32768
