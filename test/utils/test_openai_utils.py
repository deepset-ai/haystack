import pytest
from unittest.mock import patch

import pytest
from tenacity import wait_none

from haystack.errors import OpenAIError, OpenAIRateLimitError, OpenAIUnauthorizedError
from haystack.utils.openai_utils import openai_request, _openai_text_completion_tokenization_details


@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt_default():
    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="text-ada-001")
    assert tokenizer_name == "r50k_base"
    assert max_tokens_limit == 2049


@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt_davinci():
    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="text-davinci-003")
    assert tokenizer_name == "p50k_base"
    assert max_tokens_limit == 4097


@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt3_5_azure():
    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="gpt-35-turbo")
    assert tokenizer_name == "cl100k_base"
    assert max_tokens_limit == 4096


@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt3_5():
    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="gpt-3.5-turbo")
    assert tokenizer_name == "cl100k_base"
    assert max_tokens_limit == 4096


@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt_4():
    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="gpt-4")
    assert tokenizer_name == "cl100k_base"
    assert max_tokens_limit == 8192


@pytest.mark.unit
def test_openai_text_completion_tokenization_details_gpt_4_32k():
    tokenizer_name, max_tokens_limit = _openai_text_completion_tokenization_details(model_name="gpt-4-32k")
    assert tokenizer_name == "cl100k_base"
    assert max_tokens_limit == 32768


@pytest.mark.unit
@patch("haystack.utils.openai_utils.requests")
def test_openai_request_retries_generic_error(mock_requests):
    mock_requests.request.return_value.status_code = 418

    with pytest.raises(OpenAIError):
        # We need to use a custom wait amount otherwise the test would take forever to run
        # as the original wait time is exponential
        openai_request.retry_with(wait=wait_none())(url="some_url", headers={}, payload={}, read_response=False)

    assert mock_requests.request.call_count == 5


@pytest.mark.unit
@patch("haystack.utils.openai_utils.requests")
def test_openai_request_retries_on_rate_limit_error(mock_requests):
    mock_requests.request.return_value.status_code = 429

    with pytest.raises(OpenAIRateLimitError):
        # We need to use a custom wait amount otherwise the test would take forever to run
        # as the original wait time is exponential
        openai_request.retry_with(wait=wait_none())(url="some_url", headers={}, payload={}, read_response=False)

    assert mock_requests.request.call_count == 5


@pytest.mark.unit
@patch("haystack.utils.openai_utils.requests")
def test_openai_request_does_not_retry_on_unauthorized_error(mock_requests):
    mock_requests.request.return_value.status_code = 401

    with pytest.raises(OpenAIUnauthorizedError):
        # We need to use a custom wait amount otherwise the test would take forever to run
        # as the original wait time is exponential
        openai_request.retry_with(wait=wait_none())(url="some_url", headers={}, payload={}, read_response=False)

    assert mock_requests.request.call_count == 1


@pytest.mark.unit
@patch("haystack.utils.openai_utils.requests")
def test_openai_request_does_not_retry_on_success(mock_requests):
    mock_requests.request.return_value.status_code = 200
    # We need to use a custom wait amount otherwise the test would take forever to run
    # as the original wait time is exponential
    openai_request.retry_with(wait=wait_none())(url="some_url", headers={}, payload={}, read_response=False)

    assert mock_requests.request.call_count == 1
